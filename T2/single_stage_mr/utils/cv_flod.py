import random


def get_5_fold(train_pos, train_neg):
    random.shuffle(train_pos)
    random.shuffle(train_neg)
    pos_len = int(len(train_pos) / 5)
    neg_len = int(len(train_neg) / 5)
    pos = []
    neg = []
    for i in range(5):
        pos.append(train_pos[i * pos_len:(i + 1) * pos_len])
        neg.append(train_neg[i * neg_len:(i + 1) * neg_len])
    return pos, neg
