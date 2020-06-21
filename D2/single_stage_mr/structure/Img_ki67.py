import cv2
import numpy as np
import torch
import single_stage_mr.utils.blurs as blurs


class Img_ki67():
    def __init__(self, infor_dict):
        self.infor = infor_dict
        self.src_img = np.array(infor_dict["src_img"])
        self.mr_feature = []

    def get_class(self):
        return self.infor["class"]

    def get_id(self):
        return self.infor["id"]

    def make_channel3_resize(self, resize_length):
        # norm
        src_img = (((self.src_img - self.src_img.min()) / (self.src_img.max() - self.src_img.min())) * 128).astype(
            np.uint8)
        roi_img = src_img[self.infor["min_y"] - 3:self.infor["max_y"] + 4,
                  self.infor["min_x"] - 3:self.infor["max_x"] + 4]
        # blur
        blur_img = blurs.Sobel_blur(src_img)[self.infor["min_y"] - 3:self.infor["max_y"] + 4,
                   self.infor["min_x"] - 3:self.infor["max_x"] + 4]
        clahe_img = blurs.CLAHE_blur(roi_img, 4)
        # resize
        roi_resize_img = cv2.resize(roi_img, (resize_length, resize_length))
        blur_resize_img = cv2.resize(blur_img, (resize_length, resize_length))
        clahe_resize_img = cv2.resize(clahe_img, (resize_length, resize_length))
        # tensor
        self.transform_tensor = torch.from_numpy(np.array([roi_resize_img, blur_resize_img, clahe_resize_img])).float()

    def get_3channel_tensor(self):
        return self.transform_tensor

    def set_vgg_pool_feature(self, vgg_feat):
        self.vgg_feature = vgg_feat

    def get_vgg_pool_feature(self):
        return self.vgg_feature

    def set_icc_feature(self, icc_feature):
        self.icc_feature = icc_feature

    def get_icc_feature(self):
        return self.icc_feature

    def set_mr_feature(self, mr_feature):
        self.mr_feature = mr_feature

    def get_mr_feature(self):
        return self.mr_feature
