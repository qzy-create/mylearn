#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test datasets
of a model

author baiyu
"""

import argparse
import os

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
from datasets.load_blind_testset import load_blind_testset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, default="CIFAR100", help='datasets name')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-num_classes', type=int, default=100, help='classes num')
    parser.add_argument('-data_type', type=str, default="jet", help='jet, falsecolor')
    parser.add_argument('-dataset_dir', type=str, default="./data", help='dataset dir')
    parser.add_argument('-blind_dir', type=str, default="blind_testset", help='blind testset')
    parser.add_argument('-image_size', type=int, default=32, help='input image size')
    args = parser.parse_args()

    result_file_path = os.path.sep.join(["blind_test_result",
                                         args.dataset,
                                         args.net,
                                         os.path.split(args.weights)[0].split("/")[-1]])
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    result_file_path = os.path.sep.join([result_file_path, os.path.split(args.weights)[-1]]).replace("pth", "txt")
    result_file = open(result_file_path, "w")
    result_file.write(args.weights + "\n")

    testset = load_blind_testset("./data", args.dataset, args.blind_dir, args.image_size, args.data_type)

    args.num_classes = 2

    net = get_network(args)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    for i, data in enumerate(testset):
        image, _, image_path = data
        image = image.unsqueeze(0)
        if args.gpu:
            image = image.cuda()

        output = net(image)
        result = torch.nn.functional.softmax(output, dim=1)
        _, pred = output.topk(1)

        result_file.write(str(i).zfill(3) + " " + image_path + " " + str(pred.detach().cpu().numpy()[0][0]) +
                          " " + str(result.detach().cpu().numpy()[0][1]) + "\n")
        print(i, image_path, pred.detach().cpu().numpy()[0][0], result.detach().cpu().numpy()[0][1])

    result_file.close()
    # with torch.no_grad():
    #     for n_iter, (image, label) in enumerate(cifar100_test_loader):
    #         print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
    #
    #         if args.gpu:
    #             image = image.cuda()
    #             label = label.cuda()
    #
    #         output = net(image)
    #         _, pred = output.topk(5, 1, largest=True, sorted=True)
    #
    #         label = label.view(label.size(0), -1).expand_as(pred)
    #         correct = pred.eq(label).float()
    #
    #         #compute top 5
    #         correct_5 += correct[:, :5].sum()
    #
    #         #compute top1
    #         correct_1 += correct[:, :1].sum()
    #
    #
    # print()
    # print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    # print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))