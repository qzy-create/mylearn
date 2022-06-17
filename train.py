# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import sklearn.metrics as sm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from datasets.dataload import get_dataset


def train(epoch):
    start = time.time()
    net.train()
    correct = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        _, preds = outputs.max(1)
        correct = preds.eq(labels).sum()
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/Accuracy', correct.float() / len(labels), n_iter)

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    TP = 0.0
    TN = 0.0
    FN = 0.0
    FP = 0.0
    pred_labels = []
    pred_probs = []
    true_labels = []

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        true_labels += list(labels.detach().cpu().numpy())

        outputs = net(images)
        output_probs = torch.nn.functional.softmax(outputs, dim=1)
        output_probs = output_probs.detach().cpu().numpy()
        pred_probs += list(output_probs[:, 1])
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        pred_labels += list(preds.detach().cpu().numpy())
        correct += preds.eq(labels).sum()
        if args.dataset != 'CIFAR100':
            TP += (((preds == 1) & (labels == 1)).sum()).type(torch.FloatTensor)
            TN += (((preds == 0) & (labels == 0)).sum()).type(torch.FloatTensor)
            FN += (((preds == 0) & (labels == 1)).sum()).type(torch.FloatTensor)
            FP += (((preds == 1) & (labels == 0)).sum()).type(torch.FloatTensor)

    finish = time.time()

    # 计算混淆矩阵和分类报告
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)
    true_labels = np.array(true_labels)
    # 混合矩阵
    m = sm.confusion_matrix(true_labels, pred_labels)
    # 分类报告
    r = sm.classification_report(true_labels, pred_labels)
    # 计算ROC、AUC和PR
    fps, tps, _ = sm.precision_recall_curve(true_labels, pred_probs)
    fpr, tpr, _ = sm.roc_curve(true_labels, pred_probs)
    roc_auc = sm.roc_auc_score(true_labels, pred_probs)
    fig = plt.figure(figsize=(8, 4))
    lw = 2  # 线宽
    # PR 曲线
    plt.subplot(121)
    plt.plot(tps, fps, marker=".", color='red', label='PR curve', lw=lw)
    plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), color='navy', lw=lw, linestyle='--')
    plt.title("PR curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    # ROC曲线
    plt.subplot(122)
    plt.plot(fpr, tpr, marker=".", color='green', label="ROC curve(AUC=%.4f)" % roc_auc)
    plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), color='navy', lw=lw, linestyle='--')
    plt.title("ROC curve(AUC=%.4f)" % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    # plt.subplots_adjust(hspace=1)
    fig.savefig(os.path.sep.join([writer.logdir, "ROC curve_{}.png".format(str(epoch).zfill(3))]), dpi=600, format='png')
    plt.close(fig)

    if args.dataset != 'CIFAR100':
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    if args.gpu:
        print('GPU INFO.....')
        # print(torch.cuda.memory_summary(), end='')
        print(torch.cuda.memory_cached(), end='')
    print('Evaluating Network.....')
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Time consumed:{:.2f}s'.format(
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            precision,
            recall,
            F1,
            finish - start
        ))
    print("confusion matrix", m, sep="\n")
    print("classification report", r, sep="\n")
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/F1', F1, epoch)
        writer.add_scalar('Test/Precision', precision, epoch)
        writer.add_scalar('Test/Recall', recall, epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, default="CIFAR100", help='datasets name')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-num_classes', type=int, default=100, help='classes num')
    parser.add_argument('-data_type', type=str, default="jet", help='jet, falsecolor')
    parser.add_argument('-dataset_dir', type=str, default="./data", help='dataset dir')
    parser.add_argument('-image_size', type=int, default=32, help='input image size.')
    args = parser.parse_args()

    if args.dataset == 'CIFAR100':
        # data preprocessing:
        cifar100_training_loader = get_training_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True
        )

        cifar100_test_loader = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True
        )
        args.num_classes = 100
    else:
        num_classes, cifar100_training_loader, cifar100_test_loader = get_dataset(opts=args)
        args.num_classes = num_classes

    net = get_network(args)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.dataset, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.dataset, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, args.dataset, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    # writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        if epoch + 1 == settings.EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='last'))

    writer.close()
