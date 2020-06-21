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
            img_ki67.make_channel3_resize_1(resize_length=resize_length)
            img_ki67.make_channel3_resize_2(resize_length=resize_length)
            samples.append(img_ki67)
    return samples


def get_data_set(path, resize_length):
    train_path = path + "/inter"
    test_path = path + "/intra"
    inter_set = get_samples(path=train_path,
                            resize_length=resize_length)
    intra_set = get_samples(test_path,
                            resize_length=resize_length)
    return inter_set, intra_set
