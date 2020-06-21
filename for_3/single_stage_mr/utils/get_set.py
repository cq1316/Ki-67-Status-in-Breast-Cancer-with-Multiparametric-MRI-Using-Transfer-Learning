import os
import json
from single_stage_mr.structure.Img_ki67 import Img_ki67


def get_samples(path, dir, resize_length):
    t2_path = path + '/T2' + dir
    samples = []
    for root, dirs, files in os.walk(t2_path):
        for file in files:
            f = open(t2_path + '/' + file)
            infor_dict_t2 = json.load(f)
            f.close()
            f = open(path + '/D2' + dir + "/" + file)
            infor_dict_d2 = json.load(f)
            f.close()
            f = open(path + '/DWI' + dir + "/" + file)
            infor_dict_dwi = json.load(f)
            f.close()
            img_ki67 = Img_ki67(infor_dict_t2, infor_dict_d2, infor_dict_dwi)
            img_ki67.make_channel3_resize_t2(resize_length=resize_length)
            img_ki67.make_channel3_resize_d2(resize_length=resize_length)
            img_ki67.make_channel3_resize_dwi(resize_length=resize_length)
            samples.append(img_ki67)
    return samples


def get_data_set(path, resize_length):
    train_neg_set = get_samples(path, "/train" + "/neg",
                                resize_length=resize_length)
    train_pos_set = get_samples(path, "/train" + "/pos",
                                resize_length=resize_length)
    test_pos_set = get_samples(path, "/test" + "/pos",
                               resize_length=resize_length)
    test_neg_set = get_samples(path, "/test" + "/neg",
                               resize_length=resize_length)
    return train_neg_set, train_pos_set, test_pos_set, test_neg_set
