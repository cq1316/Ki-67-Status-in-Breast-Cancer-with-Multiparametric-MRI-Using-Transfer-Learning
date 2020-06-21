import cv2
import numpy as np
import torch
import single_stage_mr.utils.blurs as blurs


class Img_ki67():
    def __init__(self, infor_dict_t2, infor_dict_d2, infor_dict_dwi):
        self.infor_t2 = infor_dict_t2
        self.infor_d2 = infor_dict_d2
        self.infor_dwi = infor_dict_dwi
        self.src_img_t2 = np.array(infor_dict_t2["src_img"])
        self.src_img_d2 = np.array(infor_dict_d2["src_img"])
        self.src_img_dwi = np.array(infor_dict_dwi["src_img"])
        self.mr_feature = []

    def get_class(self):
        return self.infor_d2["class"]

    def get_id(self):
        return self.infor_d2["id"]

    # input for vgg
    def make_channel3_resize_t2(self, resize_length):
        # norm
        src_img = (
            ((self.src_img_t2 - self.src_img_t2.min()) / (self.src_img_t2.max() - self.src_img_t2.min())) * 255).astype(
            np.uint8)
        roi_img = src_img[self.infor_t2["min_y"] - 3:self.infor_t2["max_y"] + 4,
                  self.infor_t2["min_x"] - 3:self.infor_t2["max_x"] + 4]
        # blur
        blur_img = blurs.Sobel_blur(src_img)[self.infor_t2["min_y"] - 3:self.infor_t2["max_y"] + 4,
                   self.infor_t2["min_x"] - 3:self.infor_t2["max_x"] + 4]
        clahe_img = blurs.CLAHE_blur(roi_img, 8)
        # resize
        roi_resize_img = cv2.resize(roi_img, (resize_length, resize_length))
        blur_resize_img = cv2.resize(blur_img, (resize_length, resize_length))
        clahe_resize_img = cv2.resize(clahe_img, (resize_length, resize_length))
        # tensor
        self.transform_tensor_t2 = torch.from_numpy(
            np.array([roi_resize_img, blur_resize_img, clahe_resize_img])).float()

    def get_3channel_tensor_t2(self):
        return self.transform_tensor_t2

    def make_channel3_resize_d2(self, resize_length):
        # norm
        src_img = (
            ((self.src_img_d2 - self.src_img_d2.min()) / (self.src_img_d2.max() - self.src_img_d2.min())) * 128).astype(
            np.uint8)
        roi_img = src_img[self.infor_d2["min_y"] - 3:self.infor_d2["max_y"] + 4,
                  self.infor_d2["min_x"] - 3:self.infor_d2["max_x"] + 4]
        # blur
        blur_img = blurs.Sobel_blur(src_img)[self.infor_d2["min_y"] - 3:self.infor_d2["max_y"] + 4,
                   self.infor_d2["min_x"] - 3:self.infor_d2["max_x"] + 4]
        clahe_img = blurs.CLAHE_blur(roi_img, 4)
        # resize
        roi_resize_img = cv2.resize(roi_img, (resize_length, resize_length))
        blur_resize_img = cv2.resize(blur_img, (resize_length, resize_length))
        clahe_resize_img = cv2.resize(clahe_img, (resize_length, resize_length))
        # tensor
        self.transform_tensor_d2 = torch.from_numpy(
            np.array([roi_resize_img, blur_resize_img, clahe_resize_img])).float()

    def get_3channel_tensor_d2(self):
        return self.transform_tensor_d2

    def make_channel3_resize_dwi(self, resize_length):
        # norm
        src_img = (
            ((self.src_img_dwi - self.src_img_dwi.min()) / (
            self.src_img_dwi.max() - self.src_img_dwi.min())) * 128).astype(
            np.uint8)
        roi_img = src_img[self.infor_dwi["min_y"] - 3:self.infor_dwi["max_y"] + 4,
                  self.infor_dwi["min_x"] - 3:self.infor_dwi["max_x"] + 4]
        # blur
        blur_img = blurs.Sobel_blur(src_img)[self.infor_dwi["min_y"] - 3:self.infor_dwi["max_y"] + 4,
                   self.infor_dwi["min_x"] - 3:self.infor_dwi["max_x"] + 4]
        clahe_img = blurs.CLAHE_blur(roi_img, 1)
        # resize
        roi_resize_img = cv2.resize(roi_img, (resize_length, resize_length))
        blur_resize_img = cv2.resize(blur_img, (resize_length, resize_length))
        clahe_resize_img = cv2.resize(clahe_img, (resize_length, resize_length))
        # tensor
        self.transform_tensor_dwi = torch.from_numpy(
            np.array([roi_resize_img, blur_resize_img, clahe_resize_img])).float()

    def get_3channel_tensor_dwi(self):
        return self.transform_tensor_dwi

    # vgg feature
    def set_vgg_pool_feature_t2(self, vgg_feat):
        self.vgg_feature_t2 = vgg_feat

    def set_vgg_pool_feature_d2(self, vgg_feat):
        self.vgg_feature_d2 = vgg_feat

    def set_vgg_pool_feature_dwi(self, vgg_feat):
        self.vgg_feature_dwi = vgg_feat

    def get_vgg_pool_feature_t2(self):
        return self.vgg_feature_t2

    def get_vgg_pool_feature_d2(self):
        return self.vgg_feature_d2

    def get_vgg_pool_feature_dwi(self):
        return self.vgg_feature_dwi

    def set_icc_feature(self, icc_feature):
        self.icc_feature = icc_feature

    def get_icc_feature(self):
        return self.icc_feature

    # mr feature
    def set_mr_feature(self, mr_feature):
        self.mr_feature = mr_feature

    def get_mr_feature(self):
        return self.mr_feature
