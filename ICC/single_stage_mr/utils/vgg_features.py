import pandas as pd
import numpy as np
import math
import torch
from torch.autograd import Variable
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def get_features(vgg_model, inter_set, intra_set, batchsize):
    inter_feat_df1 = get_vgg_pool_feature_1(vgg_model, inter_set, batchsize)
    inter_feat_df2 = get_vgg_pool_feature_2(vgg_model, inter_set, batchsize)
    intra_feat_df1 = get_vgg_pool_feature_1(vgg_model, intra_set, batchsize)
    intra_feat_df2 = get_vgg_pool_feature_2(vgg_model, intra_set, batchsize)
    return inter_feat_df1, inter_feat_df2, intra_feat_df1, intra_feat_df2


def get_vgg_pool_feature_1(vgg_model, data_set, batchsize):
    batch_num = math.ceil(len(data_set) / batchsize)
    feats = []
    for n in range(0, batch_num):
        batch = []
        for img_ki67 in data_set[n * batchsize:(n + 1) * batchsize]:
            # get_crop_img 获取截图、sobel滤波图、局部直方图均衡化图 正确
            transform_tensor = img_ki67.get_3channel_tensor_1()
            data = Variable(transform_tensor.unsqueeze(0))
            batch.append(data)
        batch = torch.cat(batch).cuda()
        # 通过预训练vgg模型得到vgg的5个池化特征图 正确
        vgg_out = vgg_model(batch)
        feats.append(vgg_out.cpu().data.numpy())
    feat_df1 = pd.DataFrame(np.concatenate(feats))
    return feat_df1


def get_vgg_pool_feature_2(vgg_model, data_set, batchsize):
    batch_num = math.ceil(len(data_set) / batchsize)
    feats = []
    for n in range(0, batch_num):
        batch = []
        for img_ki67 in data_set[n * batchsize:(n + 1) * batchsize]:
            # get_crop_img 获取截图、sobel滤波图、局部直方图均衡化图 正确
            transform_tensor = img_ki67.get_3channel_tensor_2()
            data = Variable(transform_tensor.unsqueeze(0))
            batch.append(data)
        batch = torch.cat(batch).cuda()
        # 通过预训练vgg模型得到vgg的5个池化特征图 正确
        vgg_out = vgg_model(batch)
        feats.append(vgg_out.cpu().data.numpy())
    feat_df2 = pd.DataFrame(np.concatenate(feats))
    return feat_df2
