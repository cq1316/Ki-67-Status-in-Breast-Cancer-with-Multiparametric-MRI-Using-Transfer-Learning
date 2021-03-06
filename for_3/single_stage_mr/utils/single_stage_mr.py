import pandas as pd
import pymrmr
import numpy as np
import math
import torch
from torch.autograd import Variable
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def make_df(train_set):
    train_features = []
    class_one = np.array([2])
    class_zero = np.array([1])
    for img_ki67 in train_set:
        feature = img_ki67.get_icc_feature()
        categry = img_ki67.get_class()
        if categry == 1:
            class_feature = np.insert(arr=feature, obj=0, values=class_one, axis=0)
        else:
            class_feature = np.insert(arr=feature, obj=0, values=class_zero, axis=0)
        train_features.append(class_feature)
    feature_len = train_features[0].size - 1
    df_names = []
    df_names.append("class")
    for i in range(feature_len):
        df_names.append("%d" % i)
    df = pd.DataFrame(data=train_features, columns=df_names)
    return df


def get_maxrel_feature(dataframe, num_features, mode="MIQ"):
    feature_index = pymrmr.mRMR(dataframe, mode, num_features)
    important_feature_index = [int(x) for x in feature_index]
    return important_feature_index


def icc_feature_filter(data_set, T2_icc_inds, D2_icc_inds, DWI_icc_inds):
    for img_ki67 in data_set:
        # 获取vgg特征
        icc_feature = []
        feature = img_ki67.get_vgg_pool_feature_t2()
        # 根据重要性特征的下标选取最重要的特征
        for i in T2_icc_inds:
            icc_feature.append(feature[i])
        feature = img_ki67.get_vgg_pool_feature_d2()
        for i in D2_icc_inds:
            icc_feature.append(feature[i])
        feature = img_ki67.get_vgg_pool_feature_dwi()
        for i in DWI_icc_inds:
            icc_feature.append(feature[i])
        icc_feature = np.array(icc_feature)
        img_ki67.set_icc_feature(icc_feature)


def get_feature(vgg_model, train_set, test_set, batch_size):
    get_vgg_feature(vgg_model=vgg_model, train_set=train_set, test_set=test_set, batchsize=batch_size, type='t2')
    get_vgg_feature(vgg_model=vgg_model, train_set=train_set, test_set=test_set, batchsize=batch_size, type='d2')
    get_vgg_feature(vgg_model=vgg_model, train_set=train_set, test_set=test_set, batchsize=batch_size, type='dwi')


def feature_select(train_set, test_set, num_features):
    logging.info("do mr")
    train_df = make_df(train_set=train_set)
    important_feature_index = get_maxrel_feature(train_df, num_features)
    logging.info("get train set maxrel features")
    get_maxrel(data_set=train_set, important_index=important_feature_index, types="train")
    logging.info("get test set maxrel features")
    get_maxrel(data_set=test_set, important_index=important_feature_index, types="test")


def get_maxrel(data_set, important_index, types):
    for img_ki67 in data_set:
        # 获取vgg特征
        feature = img_ki67.get_icc_feature()
        mr_feature = []
        # 根据重要性特征的下标选取最重要的特征
        for i in important_index:
            mr_feature.append(feature[i])
        mr_feature = np.array(mr_feature)
        img_ki67.set_mr_feature(mr_feature=mr_feature)


def get_vgg_feature(vgg_model, train_set, test_set, batchsize, type):
    get_vgg_pool_feature(vgg_model=vgg_model,
                         data_set=train_set,
                         batchsize=batchsize,
                         type=type)
    get_vgg_pool_feature(vgg_model=vgg_model,
                         data_set=test_set,
                         batchsize=batchsize,
                         type=type)


def get_vgg_pool_feature(vgg_model, data_set, batchsize, type):
    batch_num = math.ceil(len(data_set) / batchsize)
    for n in range(0, batch_num):
        batch = []
        for img_ki67 in data_set[n * batchsize:(n + 1) * batchsize]:
            # get_crop_img 获取截图、sobel滤波图、局部直方图均衡化图 正确
            transform_tensor = 0
            if type == 't2':
                transform_tensor = img_ki67.get_3channel_tensor_t2()
            elif type == 'd2':
                transform_tensor = img_ki67.get_3channel_tensor_d2()
            elif type == 'dwi':
                transform_tensor = img_ki67.get_3channel_tensor_dwi()
            data = Variable(transform_tensor.unsqueeze(0))
            batch.append(data)
        batch = torch.cat(batch).cuda()
        # 通过预训练vgg模型得到vgg的5个池化特征图 正确
        vgg_out = vgg_model(batch)
        # 设置vgg_featuer 正确
        for img_ki67, vgg_pool_feat in zip(data_set[n * batchsize:(n + 1) * batchsize], vgg_out.cpu().data.numpy(), ):
            if type == 't2':
                img_ki67.set_vgg_pool_feature_t2(vgg_pool_feat)
            elif type == 'd2':
                img_ki67.set_vgg_pool_feature_d2(vgg_pool_feat)
            elif type == 'dwi':
                img_ki67.set_vgg_pool_feature_dwi(vgg_pool_feat)
