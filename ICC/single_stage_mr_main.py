# -*- coding: utf-8 -*
from single_stage_mr.utils.get_set import get_data_set
from single_stage_mr.models.feature_extracter import PoolVgg
from single_stage_mr.utils.vgg_features import get_features
import sys
import logging
import os
import pyicc
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

if __name__ == "__main__":
    path = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    logging.info("get data set")
    inter_set, intra_set = get_data_set(path=path,
                                        resize_length=80)
    # vgg特征提取器
    vgg_feature_selector = PoolVgg().cuda()
    inter_feat_df1, inter_feat_df2, intra_feat_df1, intra_feat_df2 = get_features(vgg_model=vgg_feature_selector,
                                                                                  inter_set=inter_set,
                                                                                  intra_set=intra_set, batchsize=16)
    print(inter_feat_df1.shape)
    print(inter_feat_df2.shape)
    feat_ind1 = pyicc._icc([inter_feat_df1, inter_feat_df2], "icc3", 0.75)
    print(len(feat_ind1))
    feat_ind2 = pyicc._icc([intra_feat_df1, intra_feat_df2], "icc3", 0.75)
    print(len(feat_ind2))
    inds = list(set(feat_ind1).intersection(set(feat_ind2)))
    print(len(inds))
    f = open(sys.argv[3], 'w', encoding="utf-8")
    d = {"feature_index": inds}
    json.dump(d, f)
    f.close()
