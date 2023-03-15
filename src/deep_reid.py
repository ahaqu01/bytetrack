import numpy as np
import yaml
import cv2
import torch
from .models.tracker.byte_tracker import BYTETracker as btracker
from .models.feature_extracter import FeatureExtraction


class DeepReid(object):
    def __init__(self,
                 extractor_config="",
                 extractor_weights="",
                 tracker_config="",
                 device=None):

        with open(extractor_config, 'r', encoding='utf-8') as ex_f:
            cont = ex_f.read()
            ex_cfg = yaml.load(cont)
            self.ex_cfg = ex_cfg
        with open(tracker_config, 'r', encoding='utf-8') as tk_f:
            cont = tk_f.read()
            tk_cfg = yaml.load(cont)
            self.tk_cfg = tk_cfg

        # create feature extractor and load weights
        print("creating person feature extactor")
        self.extractor = FeatureExtraction(ex_cfg, device=device)
        print("loading person feature extactor weights")
        extractor_weights_state_dict = \
            torch.load(extractor_weights, map_location=lambda storage, loc: storage.cuda(device))["model"]
        incompatible = self.extractor.load_state_dict(extractor_weights_state_dict, strict=False)
        if incompatible.missing_keys:
            print("missing_keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("unexpected_keys:", incompatible.unexpected_keys)
        print("person feature extactor weights loaded")

        # create tracker
        print("creating person tracker")
        self.track_thresh = self.tk_cfg["byte_tracker"]["track_thresh"]
        self.det_thresh = self.tk_cfg["byte_tracker"]["det_thresh"]
        self.max_time_lost = self.tk_cfg["byte_tracker"]["max_time_lost"]
        self.match_thresh = self.tk_cfg["byte_tracker"]["match_thresh"]
        self.aspect_ratio_thresh = self.tk_cfg["byte_tracker"]["aspect_ratio_thresh"]
        self.min_box_area = self.tk_cfg["byte_tracker"]["min_box_area"]
        self.out_tresh = self.tk_cfg["byte_tracker"]["out_tresh"]
        self.tracker = btracker(self.track_thresh, self.det_thresh, self.max_time_lost, self.match_thresh)
        print("person tracker created")

        self.tracker_ids_all = set()

    def reset(self):
        self.tracker.reset(self.track_thresh, self.det_thresh, self.max_time_lost, self.match_thresh)
        self.tracker_ids_all = set()

    @torch.no_grad()
    def update(self, bbox_xyxy, confidences, ori_img):
        # bbox_xyxy: ndarray, (N, 4), left,top,right,bottom
        # confidences: ndarray, (N,)
        # ori_img; ndarray; BGR, (H, W, 3)
        outputs = {}
        added_track_ids = []
        if not len(confidences):
            return outputs, added_track_ids
        else:
            self.height, self.width = ori_img.shape[:2]
            output_stracks = self.tracker.update(bbox_xyxy, confidences)
            bbox_xyxy_tracker = np.array([np.array(self._tlwh_to_xyxy(track.tlwh)) for track in output_stracks])
            features, im_crops = self._get_features(bbox_xyxy_tracker, ori_img)

            for i, out_strack in enumerate(output_stracks):
                track_id = out_strack.track_id
                tlwh = out_strack.tlwh
                confidence_i = out_strack.score
                feature_i = features[i]
                person_img_i = im_crops[i]
                bbox_xyxy_i = bbox_xyxy_tracker[i]
                if tlwh[2] * tlwh[3] > self.min_box_area \
                        and not (tlwh[2] / tlwh[3] > self.aspect_ratio_thresh) \
                        and confidence_i > self.out_tresh:
                    outputs[track_id] = {"bbox": bbox_xyxy_i,
                                         "confidence": confidence_i,
                                         "feature": feature_i,
                                         "person_img": person_img_i}
                    if out_strack.track_id not in self.tracker_ids_all:
                        added_track_ids.append(out_strack.track_id)
                        self.tracker_ids_all.add(out_strack.track_id)
            return outputs, added_track_ids

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        extractor_input = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im_crop = ori_img[int(y1):int(y2), int(x1):int(x2)]
            im_crop = im_crop[:, :, ::-1]  # reid 前处理 bgr->rgb
            im_crops.append(im_crop)
            resized_im_crop = cv2.resize(im_crop, (128, 256), interpolation=cv2.INTER_CUBIC)
            extractor_input.append(torch.as_tensor(resized_im_crop.astype("float32").transpose(2, 0, 1))[None])
        if len(extractor_input) > 0:
            batch_image = torch.cat(extractor_input, dim=0)
            features = self.extractor(batch_image).numpy()
        else:
            features = np.array([])
        return features, im_crops
