import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - 'causal', 'interval', or 'order'.
            train_cls - this is for Stark-ST, should be False for SeqTrack.

        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """

        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        # sample a sequence from the given dataset
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

        if is_video_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            template_frame_ids, search_frame_ids = self.sequential_sample(visible)

        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames

        template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        data = TensorDict({'template_images': np.array(template_frames),
                           'template_anno': np.array([bbox.numpy() for bbox in template_anno['bbox']]),
                           'search_images': np.array(search_frames),
                           'search_anno': np.array([bbox.numpy() for bbox in search_anno['bbox']]),
                           'dataset': dataset.get_name(),
                           'num_frames': len(search_frames),
                           'test_class': meta_obj_test.get('object_class_name')})
        # self.show(data, 'search', 0)

        return data

    def show(self, data, strr, i):
        image = data[strr + '_images'][i]
        image.transpose(2, 0, 1)
        _, H, W = image.shape
        import cv2
        x1, y1, w, h = data[strr + '_anno'][i]
        x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)
        image_show = image
        image_show = np.ascontiguousarray(image_show.astype('uint8'))
        cv2.rectangle(image_show, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        cv2.imshow(strr + str(i), image_show)
        if cv2.waitKey() & 0xFF == ord('q'):
            pass

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        # add by chenxin to debug
        count = 0
        seq_id, visible, seq_info_dict = None, None, None
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
            count += 1
            if count > 200:
                print("too large count")
                print(str(count))
        return seq_id, visible, seq_info_dict

    def sequential_sample(self, visible):
        # Sample frames in sequential manner
        template_frame_ids = self._sample_visible_ids(visible, num_ids=1, min_id=0,
                                                      max_id=len(visible) - self.num_search_frames)
        if self.max_gap == -1:
            left = template_frame_ids[0]
        else:
            # template frame (1) ->(max_gap) -> search frame (num_search_frames)
            left_max = min(len(visible) - self.num_search_frames, template_frame_ids[0] + self.max_gap)
            left = self._sample_visible_ids(visible, num_ids=1, min_id=template_frame_ids[0],
                                            max_id=left_max)[0]

        valid_ids = [i for i in range(left, len(visible)) if visible[i]]
        search_frame_ids = valid_ids[:self.num_search_frames]

        # if length is not enough
        last = search_frame_ids[-1]
        while len(search_frame_ids) < self.num_search_frames:
            if last >= len(visible) - 1:
                search_frame_ids.append(last)
            else:
                last += 1
                if visible[last]:
                    search_frame_ids.append(last)

        return template_frame_ids, search_frame_ids
