# -*- coding: utf-8 -*
import json
import logging
import math
import os
import random
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable

from single_stage_mr.models.feature_extracter import PoolVgg
from single_stage_mr.models.mlp import MLPClassifier
from single_stage_mr.utils.cv_flod import get_5_fold
from single_stage_mr.utils.get_set import get_data_set
from single_stage_mr.utils.single_stage_mr import icc_feature_filter, get_vgg_feature, feature_select

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def train_batch(epoch, train_set, optimizer, model, batchsize):
    batch_num = math.ceil(len(train_set) / batchsize)
    train_loss = 0
    model.train(mode=True)
    for n in range(0, batch_num):
        targets = []
        zscore = []
        for img_ki67 in train_set[n * batchsize:(n + 1) * batchsize]:
            zscore_feature = img_ki67.get_mr_feature()
            zscore_feature = Variable(torch.FloatTensor(zscore_feature).unsqueeze(0))
            zscore.append(zscore_feature)
            target = img_ki67.get_class()
            targets.append(target)
        zscore = torch.cat(zscore).cuda()
        targets = Variable(torch.LongTensor(targets)).cuda()
        optimizer.zero_grad()
        output = model(zscore)
        loss = F.cross_entropy(input=output,
                               target=targets,
                               weight=torch.FloatTensor([0.7, 0.3]).cuda())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    # logging.info('--------------------------------')
    # logging.info('Train Epoch: %d loss : %f' % (epoch, train_loss / len(train_set)))
    return train_loss / len(train_set)


def test(test_set, model):
    y_t = []
    y_p = []
    model.eval()
    for img_ki67 in test_set:
        # 取出mrmr特征
        zscore_feature = img_ki67.get_mr_feature()
        zscore_feature = Variable(torch.FloatTensor(zscore_feature).unsqueeze(0)).cuda()
        target = img_ki67.get_class()
        y_t.append(target)
        output = model(zscore_feature)
        y_p.append(output.data[0].cpu()[1])
    return y_t, y_p


def evaluate_on_trainset(fpr, tpr, threshold, pos_num, neg_num):
    '''find cutoff'''
    temp_pos = 0
    distance = 0
    for i in range(len(fpr)):
        if distance < (tpr[i] - fpr[i]) / math.sqrt(2):
            distance = (tpr[i] - fpr[i]) / math.sqrt(2)
            temp_pos = i
    cutoff = threshold[temp_pos]
    TP = tpr[temp_pos] * pos_num
    TN = (1 - fpr[temp_pos]) * neg_num
    FP = neg_num - TN
    FN = pos_num - TP
    # sen = tpr[temp_pos]
    # spe = -fpr[temp_pos] + 1
    acc = (TP + TN) / (FP + FN + TP + TN)
    # ppv = TP / (TP + FP + 0.001)
    # npv = TN / (TN + FN + 0.001)
    # logging.info("cutoff:%f" % cutoff)
    # logging.info("tp:%d tn:%d fp:%d fn:%d" % (TP, TN, FP, FN))
    # logging.info('SEN:%f SPE:%f PPV:%f NPV:%f Acc:%f' % (sen, spe, ppv, npv, acc))
    return acc, TP, TN, FP, FN, cutoff


def evaluate_on_testset(fpr, tpr, threshold, cutoff, pos_num, neg_num):
    '''find tpr and fpr'''
    temp_pos = 0
    for cont, thre in enumerate(threshold):
        if thre < cutoff:
            temp_pos = cont
            break
    if temp_pos == 0:
        temp_pos = 1
        print('temp_pos is 0')
    proportion = (threshold[temp_pos - 1] - cutoff) / (threshold[temp_pos - 1] - threshold[temp_pos])
    sen = tpr[temp_pos - 1] + (tpr[temp_pos] - tpr[temp_pos - 1]) * proportion
    spe = 1 - (fpr[temp_pos - 1] + (fpr[temp_pos] - fpr[temp_pos - 1]) * proportion)
    TP = sen * pos_num
    TN = spe * neg_num
    FP = neg_num - TN
    FN = pos_num - TP
    acc = (TP + TN) / (FP + FN + TP + TN)
    # ppv = TP / (TP + FP + 0.001)
    # npv = TN / (TN + FN + 0.001)
    # logging.info("cutoff:%f" % cutoff)
    # logging.info("tp:%d tn:%d fp:%d fn:%d" % (TP, TN, FP, FN))
    # logging.info('SEN:%f SPE:%f PPV:%f NPV:%f Acc:%f' % (sen, spe, ppv, npv, acc))
    return acc, TP, TN, FP, FN


if __name__ == "__main__":
    path = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    logging.info("get data set")
    train_neg_set, train_pos_set, test_pos_set, test_neg_set = get_data_set(path=path,
                                                                            resize_length=80)
    f = open(sys.argv[3])
    d = json.load(f)
    icc_inds = d["feature_index"]
    f.close()
    # 分数据集
    total_train_set = train_neg_set + train_pos_set
    total_test_set = test_neg_set + test_pos_set
    # 负样本的数量
    length = len(train_neg_set)
    logging.info("get data set completed")
    logging.info("get feature and cv")
    # vgg特征提取器
    vgg_feature_selector = PoolVgg().cuda()
    # 混合mr
    get_vgg_feature(vgg_model=vgg_feature_selector,
                    train_set=total_train_set,
                    test_set=total_test_set,
                    batchsize=16)
    icc_feature_filter(data_set=total_train_set + total_test_set, icc_inds=icc_inds)
    pos, neg = get_5_fold(train_pos=train_pos_set,
                          train_neg=train_neg_set)
    cols = ["train_tp", "train_tn", "train_fp", "train_fn", "train_acc", "train_auc",
            "val_tp", "val_tn", "val_fp", "val_fn", "val_acc", "val_auc",
            "test_tp", "test_tn", "test_fp", "test_fn", "test_acc", "test_auc",
            "cutoff", "feature_num", "hidden_num"]
    experiment_df = pd.DataFrame(columns=cols)
    for feature_num in range(20, 600, 20):
        for hidden_num in [16, 32, 64, 96, 128]:
            metrics = [0 for i in range(19)]
            for val_ind in range(5):
                inds = list(range(5))
                inds.remove(val_ind)
                train_pos = []
                train_neg = []
                val_pos = pos[val_ind]
                val_neg = neg[val_ind]
                for ind in inds:
                    train_pos = train_pos + pos[ind]
                    train_neg = train_neg + neg[ind]
                train_set = train_pos + train_neg
                val_set = val_pos + val_neg
                feature_select(train_set=train_set,
                               test_set=val_set + total_test_set,
                               num_features=feature_num)
                for i in range(3):
                    lr = 0.001
                    model = MLPClassifier(num_class=2, num_feature=feature_num, num_hidden=hidden_num).cuda()
                    classifier_optimizer = optim.Adam(model.parameters(),
                                                      lr=lr,
                                                      weight_decay=5e-4)
                    runs = 20
                    errors = [100 for j in range(runs)]
                    accs = [0 for j in range(runs)]
                    pre_error = 0
                    change_lr = False
                    pre_acc = 0
                    for epoch in range(1, 1000):
                        if change_lr:
                            for param_group in classifier_optimizer.param_groups:
                                param_group['lr'] = lr
                        random.shuffle(train_set)
                        error = train_batch(epoch=epoch,
                                            train_set=train_set,
                                            optimizer=classifier_optimizer,
                                            model=model,
                                            batchsize=16)
                        # 记训练集的情况
                        # logging.info("training set")
                        y_t1, y_p1 = test(test_set=train_set, model=model)
                        fpr_tr, tpr_tr, threshold = roc_curve(y_t1, y_p1, pos_label=1)
                        roc_auc_tr = auc(fpr_tr, tpr_tr)
                        # logging.info('auc:%f' % roc_auc_tr)
                        acc_tr, TP_tr, TN_tr, FP_tr, FN_tr, cutoff = evaluate_on_trainset(fpr=fpr_tr,
                                                                                          tpr=tpr_tr,
                                                                                          threshold=threshold,
                                                                                          pos_num=len(train_pos),
                                                                                          neg_num=len(train_neg))
                        # 记测试集的情况
                        # logging.info("val set")
                        y_t2, y_p2 = test(test_set=val_set, model=model)
                        fpr_te, tpr_te, threshold = roc_curve(y_t2, y_p2, pos_label=1)
                        roc_auc_val = auc(fpr_te, tpr_te)
                        # logging.info('auc:%f' % roc_auc_val)
                        acc_val, TP_val, TN_val, FP_val, FN_val = evaluate_on_testset(fpr=fpr_te,
                                                                                      tpr=tpr_te,
                                                                                      threshold=threshold,
                                                                                      cutoff=cutoff,
                                                                                      pos_num=len(val_pos),
                                                                                      neg_num=len(val_neg))

                        acc_mean = sum(accs) / runs
                        if (round(acc_tr, ndigits=3) == round(acc_mean, ndigits=3)
                            and round(acc_tr, ndigits=3) == round(pre_acc, ndigits=3)) or epoch == 500:
                            # logging.info("test set")
                            y_t2, y_p2 = test(test_set=total_test_set, model=model)
                            fpr_te, tpr_te, threshold = roc_curve(y_t2, y_p2, pos_label=1)
                            roc_auc_te = auc(fpr_te, tpr_te)
                            # logging.info('auc:%f' % roc_auc_te)
                            acc_te, TP_te, TN_te, FP_te, FN_te = evaluate_on_testset(fpr=fpr_te,
                                                                                     tpr=tpr_te,
                                                                                     threshold=threshold,
                                                                                     cutoff=cutoff,
                                                                                     pos_num=len(test_pos_set),
                                                                                     neg_num=len(test_neg_set))
                            m = [acc_tr, TP_tr, TN_tr, FP_tr, FN_tr, roc_auc_tr,
                                 acc_val, TP_val, TN_val, FP_val, FN_val, roc_auc_val,
                                 acc_te, TP_te, TN_te, FP_te, FN_te, roc_auc_te, cutoff]
                            metrics = [metrics[i] + m[i] for i in range(19)]
                            break
                        else:
                            accs.pop(0)
                            accs.append(acc_tr)
                            pre_acc = acc_tr
                        error_mean = sum(errors) / runs
                        if abs(error - pre_error) < 0.001 and abs(error - error_mean) < 0.001:
                            lr = lr * 0.6
                            change_lr = True
                            errors = [100 for j in range(runs)]
                        else:
                            change_lr = False
                        errors.pop(0)
                        errors.append(error)
                        pre_error = error
            metrics = [metrics[i] / 15 for i in range(19)]
            metrics = metrics + [feature_num, hidden_num]
            experiment_data = pd.DataFrame(data=[metrics], columns=cols)
            experiment_df = experiment_df.append(experiment_data)
    experiment_df.to_csv("experiments/experiment_%d.csv" %3, index=0)
