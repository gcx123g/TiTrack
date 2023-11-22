import cv2
import numpy as np

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
import lib.train.data.bounding_box_utils as bbutils


def get_subwindow(img, pos, model_sz, original_sz, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    sz = original_sz
    im_sz = img.shape
    c = (sz + 1) / 2

    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1

    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = img.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = img
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = img[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    im_patch = cv2.resize(im_patch, (model_sz, model_sz))

    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    im_patch = im_patch.cuda()
    return im_patch


def show(image, bbox, sz):
    a = image[0].permute(1, 2, 0)
    import cv2
    x, y, w, h = int(128 - (bbox[2] * 256 / sz) / 2), int(128 - (bbox[3] * 256 / sz) / 2), \
                 int(bbox[2] * 256 / sz), int(bbox[3] * 256 / sz)
    image_show = a.cpu().numpy()
    image_show = np.ascontiguousarray(image_show.astype('uint8'))
    cv2.rectangle(image_show, (x, y, w, h), color=(0, 0, 255), thickness=2)
    cv2.imshow('i', image_show)
    if cv2.waitKey() & 0xFF == ord('q'):
        pass

def show_gray(image, bbox, sz):
    a = image.permute(1, 2, 0)
    import cv2
    x, y, w, h = int(128 - (bbox[2] * 256 / sz) / 2), int(128 - (bbox[3] * 256 / sz) / 2), \
                 int(bbox[2] * 256 / sz), int(bbox[3] * 256 / sz)
    image_show = a.cpu().numpy()
    image_show = np.ascontiguousarray(image_show.astype('uint8'))
    cv2.rectangle(image_show, (x, y, w, h), color=(0, 0, 255), thickness=2)
    cv2.imshow('i', image_show)
    if cv2.waitKey() & 0xFF == ord('q'):
        pass


def bbox_norm(search_anno, search_image):
    b, n, _ = search_anno.shape
    bbox = []
    for idx in range(b):
        w, h = search_image[idx].shape[-2, -1]
        bbox_temp = search_anno[idx]
    return 0


class SeqTrackActor(BaseActor):
    """ Actor for training the SeqTrack"""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.BINS = cfg.MODEL.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT
        self.num_frames = cfg.DATA.SEARCH.NUMBER
        self.template_size = cfg.DATA.TEMPLATE.SIZE
        self.template_factor = cfg.DATA.TEMPLATE.FACTOR
        self.center = []
        self.s_z = None
        self.search_size = cfg.DATA.SEARCH.SIZE
        self.search_factor = cfg.DATA.SEARCH.FACTOR
        self.search_center_jitter = cfg.DATA.SEARCH.CENTER_JITTER
        self.search_scale_jitter = cfg.DATA.SEARCH.SCALE_JITTER

    def tem_crop(self, images, bboxs):
        batch_size = len(images)
        z_crop = []
        for idx in range(batch_size):
            here_template = images[idx][0]
            here_bbox = bbutils.batch_xywh2center2(bboxs[idx])[0]

            w_z = here_bbox[2] * self.template_factor
            h_z = here_bbox[3] * self.template_factor
            s_z = np.ceil(np.sqrt(w_z * h_z))

            channel_average = np.mean(here_template, axis=(0, 1))
            here_crop = get_subwindow(here_template, here_bbox[:2], self.template_size, s_z, channel_average)
            z_crop.append(here_crop)
        return z_crop

    def search_crop(self, images, bboxs):
        batch_size = len(images)
        num_frames = self.num_frames
        x_crop = []
        pre_seq = []
        for idx in range(batch_size):
            here_searchs = images[idx]
            here_bboxs = bbutils.batch_xywh2center2(bboxs[idx])
            for i in range(num_frames):
                if i == 0:
                    self.center = here_bboxs[i][:2]
                    w_z = self.center[0] * self.search_factor
                    h_z = self.center[1] * self.search_factor
                    self.s_z = np.ceil(np.sqrt(w_z * h_z))
                    channel_average = np.mean(here_searchs[i], axis=(0, 1))
                    ori_crop = get_subwindow(here_searchs[i], self.center, self.search_size, self.s_z,
                                             channel_average)
                    crop = ori_crop
                    # show(ori_crop, here_bboxs[i], self.s_z)
                else:
                    channel_average = np.mean(here_searchs[i], axis=(0, 1))
                    here_crop = get_subwindow(here_searchs[i], self.center, self.search_size, self.s_z,
                                              channel_average)
                    # show(here_crop, here_bboxs[i-1], self.s_z)
                    crop = torch.cat((crop, here_crop), 0)

                    self.center = here_bboxs[i][:2]
                    w_z = self.center[0] * self.search_factor
                    h_z = self.center[1] * self.search_factor
                    self.s_z = np.ceil(np.sqrt(w_z * h_z))

                    if i == 1:
                        temp_seq = torch.mean(crop[i]-crop[i-1], dim=0, keepdim=True)
                    else:
                        temp_seq = torch.cat((temp_seq, torch.mean(crop[i]-crop[i-1], 0, keepdim=True)), 0)
                        temp_seq = torch.cat((temp_seq, torch.mean(temp_seq[i-1]-crop[i-2], 0, keepdim=True)), 0)
                        # show_gray(temp_seq[0].unsqueeze(0), here_bboxs[i - 1], self.s_z)
                        # show_gray(temp_seq[1].unsqueeze(0), here_bboxs[i - 1], self.s_z)
                        # show_gray(temp_seq[2].unsqueeze(0), here_bboxs[i - 1], self.s_z)
            pre_seq.append(temp_seq.unsqueeze(0))
            x_crop.append(crop)
        return x_crop, pre_seq

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, targets = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(outputs, targets)

        return loss, status

    def forward_pass(self, data):
        n, b = data['num_frames'][0], len(data['search_images'])
        template_images, template_annos = data['template_images'], data['template_anno']
        search_images, search_annos = data['search_images'], data['search_anno']

        template_anno = np.array([tem_anno for tem_anno in template_annos])
        search_anno = np.array([search_anno for search_anno in search_annos])

        # crop list[b*, n, c, h, w]
        z_list = self.tem_crop(template_images, template_anno)
        x_list, seq_list = self.search_crop(search_images, search_anno)
        bbox_list = [torch.from_numpy(search_bbox)[-1] for search_bbox in data['search_anno']]

        feature_xz = self.net(z_list, x_list, mode='encoder')  # forward the encoder

        outputs = self.net(xz=feature_xz, seq=seq_list, mode="decoder")
        targets = torch.stack(bbox_list, 0)

        return outputs, targets

    def compute_losses(self, outputs, targets_seq, return_status=True):
        # Get loss
        ce_loss = self.objective['ce'](outputs, targets_seq)
        # weighted sum
        loss = self.loss_weight['ce'] * ce_loss

        outputs = outputs.softmax(-1)
        outputs = outputs[:, :self.BINS]
        value, extra_seq = outputs.topk(dim=-1, k=1)
        boxes_pred = extra_seq.squeeze(-1).reshape(-1, 5)[:, 0:-1]
        boxes_target = targets_seq.reshape(-1, 5)[:, 0:-1]
        boxes_pred = box_cxcywh_to_xyxy(boxes_pred)
        boxes_target = box_cxcywh_to_xyxy(boxes_target)
        iou = box_iou(boxes_pred, boxes_target)[0].mean()

        if return_status:
            # status for log
            status = {"Loss/total": loss.item(),
                      "IoU": iou.item()}
            return loss, status
        else:
            return loss

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)
