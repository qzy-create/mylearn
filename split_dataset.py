# -*- coding: utf-8 -*-
import os
import numpy as np

from imutils import paths

np.random.seed(49)

if __name__ == '__main__':
    train_set = "train.txt"
    test_set = "test.txt"
    val_set = "blind_test.txt"

    train_file = open(train_set, "w")
    test_file = open(test_set, "w")
    val_file = open(val_set, "w")

    # 训练集和测试集比例
    rate = 0.7

    T_sample = np.array(sorted(paths.list_files("./T_labels", validExts=".xml")))
    F_sample = np.array(sorted(paths.list_files("./F_labels", validExts=".xml")))
    T_sample = T_sample.reshape((T_sample.size, -1))
    F_sample = F_sample.reshape((F_sample.size, -1))
    np.random.shuffle(T_sample)
    np.random.shuffle(F_sample)
    # T_sample = T_sample[:1000]
    # F_sample = F_sample[:1000]

    train_set_num = int(250 * rate)
    for i, T_path in enumerate(T_sample):
        T_id = (os.path.split(T_path[0])[-1]).split(".")[0]
        F_id = (os.path.split(F_sample[i][0])[-1]).split(".")[0]
        if i < train_set_num:
            train_file.write("{} 0\n".format(F_id))
            train_file.write("{} 1\n".format(T_id))
        elif train_set_num <= i < 225:
            test_file.write("{} 0\n".format(F_id))
            test_file.write("{} 1\n".format(T_id))
        else:
            val_file.write("{} 0\n".format(F_id))
            val_file.write("{} 1\n".format(T_id))


    train_file.close()
    test_file.close()
    val_file.close()
