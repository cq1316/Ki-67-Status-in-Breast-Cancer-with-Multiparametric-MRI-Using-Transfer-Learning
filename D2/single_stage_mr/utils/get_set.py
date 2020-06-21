import os
import json
from single_stage_mr.structure.Img_ki67 import Img_ki67



def get_samples(path, resize_length):
    samples = []
    for root, dirs, files in os.walk(path):
        for file in files:
            f = open(root + "/" + file)
            infor_dict = json.load(f)
            f.close()
            img_ki67 = Img_ki67(infor_dict)
            img_ki67.make_channel3_resize(resize_length=resize_length)
            samples.append(img_ki67)
    return samples


def get_data_set(path, resize_length):
    train_path = path + "/train"
    test_path = path + "/test"
    train_neg_set = get_samples(path=train_path + "/neg",
                                resize_length=resize_length)
    train_pos_set = get_samples(train_path + "/pos",
                                resize_length=resize_length)
    test_pos_set = get_samples(test_path + "/pos",
                               resize_length=resize_length)
    test_neg_set = get_samples(test_path + "/neg",
                               resize_length=resize_length)
    return train_neg_set, train_pos_set, test_pos_set, test_neg_set
