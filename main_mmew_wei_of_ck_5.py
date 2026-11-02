import argparse
import os
import yaml
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import utils
import time
import sys
import random
from datetime import datetime
from datasets.ck import CK_five
from datasets.mmew_wei import MMEW_wei_five
from sklearn.metrics import confusion_matrix
from models.transfer_backbone import Transfer_Network
import torch.backends.cudnn as cudnn
from models.IDE_block import DPGN
from torchmetrics import MetricCollection, Accuracy, F1Score, Recall, Precision
from models.utils_functions import initialize_nodes_edges, OT, MMD_loss
from tqdm import tqdm
from utils import *
from train_eval_utils import *



def main(config):
    set_global_seed(42)
    # 设置评价指标
    svname = '{}-{}-{}'.format(config['train_macro_dataset'], config['train_micro_dataset'], config['num_class'])
    svname_tb = svname + '_' + config['model']
    svname_pth = svname + '_pth'

    save_path_tb = os.path.join(config['save'], svname_tb)
    save_path_pth = os.path.join(config['save'], svname_pth)

    utils.ensure_path(save_path_tb)
    utils.set_log_path(save_path_tb)
    utils.ensure_path(save_path_pth)

    logger = setup_logging(save_path_tb)
    logger.info(f"Save path: {save_path_tb}")  # logging的保存路径和tensorboard一样
    print(f'Start Tensorboard with "tensorboard --logdir={save_path_tb}", view at http://localhost:6006/')
    writer = SummaryWriter(os.path.join(save_path_tb, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path_tb, 'config.yaml'), 'w'))
    '''
    # 设置日志路径
    # set_log_path(save_path_tb)  # 替换为实际的路径
    # 记录日志
    # log("This is a test log message.")
    # log("Here is another log entry.")
    '''

    file_end = os.path.join(config['save'],
                            "{}_result_{}.txt".format(config['train_micro_dataset'], config['num_class']))

    fp_end = open(file_end, "w")
    print("hello!", file=fp_end)

    #### Dataset ####

    # 按照loso的划分方法分别写入train和test相应的txt文件并保存
    LOSO = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
            'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
    # LOSO = ['S30']
    #
    # 创建两个个list分别存储真实标签和预测标签，这里的预测标签是每一个subject的预测标签拼接起来的，一个subject跑100个epoch
    ture_all_list = []
    pred_all_list = []
    best_all_acc, best_all_rec, best_all_pre, best_all_f_1 = 0, 0, 0, 0
    start_time = time.time()
    count_sub = 0
    # LOSO 开始循环
    for i in range(len(LOSO)):
        print(LOSO[i])
        logger.info(f"sub: {LOSO[i]}")
        file_name = os.path.join(config['save'],
                                 "{}_result_{}_{}.txt".format(config['train_micro_dataset'], config['num_class'], LOSO[i]))
        fp_name = open(file_name, "w")
        print("hello!", file=fp_name)
        ## data augmentation for training set only
        data_transforms_norm = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(4),
            transforms.RandomCrop(224, padding=4),
        ])
        # data_transforms_norm = None
        data_transform = {
            "train": transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Lambda(lambd=lambda x: x / 255),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.2),
                # transforms.RandomErasing(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            ]),
            "valid": transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Lambda(lambd=lambda x: x / 255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])}
        ''' Dataset'''
        train_macro_dataset = CK_five(config['train_macro_dataset'], transform=data_transform['train'], transform_norm=data_transforms_norm, **config['train_macro_dataset_args'])
        data_dict = {"sub_id": LOSO[i]}
        train_micro_dataset = MMEW_wei_five(config['train_micro_dataset'], transform=data_transform['train'], transform_norm=data_transforms_norm, **config['train_micro_dataset_args'], **data_dict)
        valid_micro_dataset = MMEW_wei_five(config['valid_micro_dataset'], transform=data_transform['valid'], **config['valid_micro_dataset_args'], **data_dict)
        # 这个地方用来跳过测试对象中测试样本数为0的情况
        sz = len(valid_micro_dataset.label)
        if not sz:
            print(f'当前的测试对象{LOSO[i]}中测试样本数为{sz}!')
            i += 1
            continue
        print('ok!')
        # 测试样本不为0
        count_sub += 1
        print(f'count_sub:{count_sub}')
        # 宏表情训练集
        train_macro_loader = DataLoader(train_macro_dataset, batch_size=20,
                                  num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
        # 微表情训练集
        train_micro_loader = DataLoader(train_micro_dataset, batch_size=20,
                                        num_workers=16, shuffle=True, drop_last=True, pin_memory=True)

        # 微表情验证集
        valid_micro_loader = DataLoader(valid_micro_dataset, batch_size=20,
                                        num_workers=16, drop_last=False, pin_memory=True)

        print('完成数据dataset和dataloader的载入！')

        ''' Model and Optimizer'''
        # 用来提取特征
        feature_encoder = Transfer_Network(numOfLayer=10, num_class=config['num_class'])
        print(feature_encoder)
        # 图交互
        IDE_block = DPGN(dropout=0.1, num_class=config['num_class'], num_sample=200,
                         metric='l2', dim_size=384)
        # 图注意力
        # 将网络模型分别放到GPU上
        # feature_encoder = torch.nn.DataParallel(feature_encoder)
        feature_encoder = feature_encoder.cuda()

        # IDE_block = torch.nn.DataParallel(IDE_block)
        IDE_block = IDE_block.cuda()

        # 定义优化器
        feature_encoder_optim, fea_lr_scheduler = utils.make_optimizer(
            feature_encoder.parameters(),
            config['optimizer'], **config['optimizer_args'])
        # feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=1e-2, weight_decay=1e-4)
        IDE_block_optim, ide_lr_scheduler = utils.make_optimizer(IDE_block.parameters(), config['optimizer'], **config['optimizer_args'])
        # Learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(feature_encoder_optim, 0.985)

        # 定义损失函数
        cross_loss = torch.nn.CrossEntropyLoss()
        mmd_loss = MMD_loss(kernel_type='linear')

        print('模型和优化器设置完成！')
        # 加载训练状态
        start_epoch = load_checkpoint(feature_encoder, feature_encoder_optim, scheduler, filename=None)
        max_epoch = config['max_epoch']
        # 每个人都要进行100个epoch，并且选出100个epoch中结果最好的一次，并且保存模型参数
        best_epoch, best_acc, best_rec, best_pre, best_f_1 = 0, 0, 0, 0, 0
        best_y_true, best_y_pred = [], []
        rec_loss = []
        for epoch in range(start_epoch, max_epoch):
            print('ok！')
            """train."""
            train_total_loss, train_clf_loss, train_cross_att_loss, train_domain_loss, train_metrics = \
                train_one_epoch(config=config,
                            feature_encoder=feature_encoder,
                            IDE_block=IDE_block,
                            feature_encoder_optim=feature_encoder_optim,
                            IDE_block_optim=IDE_block_optim,
                            cross_loss=cross_loss,
                            mmd_loss=mmd_loss,
                            train_macro_loader=train_macro_loader,
                            train_micro_loader=train_micro_loader,
                            epoch=epoch,
                            warmup=True)
            LoggerInfo = f"Subject = {LOSO[i]}, \tEpoch = {epoch}, \ttra_ACC = {train_metrics['ACC']:.6f}, \ttra_REC = {train_metrics['REC']:.6f}, \ttra_PRE = {train_metrics['PRE']:.6f}, \ttra_F_1 = {train_metrics['F_1']:.6f}\n" \
                         f"Total loss = {train_total_loss:.6f}, \tCE loss = {train_clf_loss:.6f}, \tMDD loss = {train_clf_loss:.6f}, \tGOT loss = {train_domain_loss:.6f}\n"
            print(LoggerInfo)
            print(LoggerInfo, file=fp_name)

            """Valid."""
            valid_clf_loss, valid_metrics, label_all, pred_all = \
                evaluate(config=config,
                     feature_encoder=feature_encoder,
                     cross_loss=cross_loss,
                     valid_micro_loader=valid_micro_loader,
                     )
            LoggerInfo = f"Subject = {LOSO[i]}, \tEpoch = {epoch}, \tval_ACC = {valid_metrics['ACC']:.6f}, \tval_REC = {valid_metrics['REC']:.6f}, \tval_PRE = {valid_metrics['PRE']:.6f}, \tval_F_1 = {valid_metrics['F_1']:.6f}\n" \
                         f"Total loss = {valid_clf_loss:.6f}"
            print(LoggerInfo)
            print(LoggerInfo, file=fp_name)

            # 每个epoch结束时，更新学习率
            print(f'Finished the epoch {epoch} training')
            param_groups = feature_encoder_optim.param_groups
            current_lr = param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")
            scheduler.step()


            # 判断每个Subject的100次epoch得到最好的一个打印并且保存模型
            # Check for best model
            is_best = valid_metrics['ACC'] > best_acc
            if is_best:
                best_acc = valid_metrics['ACC']
                best_rec = valid_metrics['REC']
                best_pre = valid_metrics['PRE']
                best_f_1 = valid_metrics['F_1']

                best_epoch = epoch
                best_y_true = label_all
                best_y_pred = pred_all

                logger.info(f"New best ACC: {best_acc:.4f}, New best epoch: {best_epoch:d}")
            # epoch, model, optimizer, scheduler, save_path, info, is_best, loso=None
            # epoch, model, optimizer, scheduler, save_path, info, is_best, loso=None
            save_checkpoint(
                epoch=epoch,
                model=feature_encoder,
                optimizer=feature_encoder_optim,
                scheduler=scheduler,
                save_path=save_path_pth,
                info=config['train_micro_dataset'] + '_' + str(config['num_class']),
                is_best=is_best,
                loso=LOSO[i])
            # if best_acc == 1:
            #     break
            print(f"[epoch {epoch}] accuracy: {valid_metrics['ACC']:.6f}")
            tags = ["loss", "accuracy", "learning_rate"]
            writer.add_scalar(tags[0], train_total_loss, epoch)
            writer.add_scalar(tags[1], valid_metrics['ACC'], epoch)
            writer.add_scalar(tags[2], feature_encoder_optim.param_groups[0]["lr"], epoch)
            # 打印真实标签和预测标签
            print(f'The true labels:{label_all}')
            print(f'The pred labels:{pred_all}')
            # 将训练结果写入txt
            print(f'The true labels:{label_all}', file=fp_name)
            print(f'The pred labels:{pred_all}', file=fp_name)
            # 保存loss数据到文件（可选）
            rec_loss.append(train_total_loss)  # 用rec_loss保存每一个epoch的loss，最后得到100个epoch的loss值
        # 保存loss数据到文件（可选） # 保存100个epoch的训练损失的loss
        loss_note = config['train_micro_dataset'] + '_' + str(config['num_class']) + '_' + LOSO[i] + 'train_losses.npy'
        np.save(os.path.join(save_path_tb, loss_note), np.array(rec_loss))
        print(
            '=================================================================================================================================>best result......')
        print(
            '=================================================================================================================================>best result......',
            file=fp_name)
        print(
            f"Subject = {LOSO[i]}\tBest_Epoch = {best_epoch},\tBest_Acc = {best_acc:.6f},\tBest_Rec = {best_rec:.6f},\tBest_Pre = {best_pre:.6f},\tBest_F_1 = {best_f_1:.6f}\n")
        print(
            f"Subject = {LOSO[i]}\tBest_Epoch = {best_epoch},\tBest_Acc = {best_acc:.6f},\tBest_Rec = {best_rec:.6f},\tBest_Pre = {best_pre:.6f},\tBest_F_1 = {best_f_1:.6f}\n",
            file=fp_name)
        print(
            f"Subject = {LOSO[i]}\tBest_Epoch = {best_epoch},\tBest_Acc = {best_acc:.6f},\tBest_Rec = {best_rec:.6f},\tBest_Pre = {best_pre:.6f},\tBest_F_1 = {best_f_1:.6f}",
            file=fp_end)
        fp_name.close()
        # 每一个人最佳的准确率加起来求平均得到最终的平均准确率
        best_all_acc = best_all_acc + best_acc
        best_all_rec = best_all_rec + best_rec
        best_all_pre = best_all_pre + best_pre
        best_all_f_1 = best_all_f_1 + best_f_1

        ture_all_list.extend(best_y_true)
        pred_all_list.extend(best_y_pred)

        # end LOSO
    ave_acc = best_all_acc / count_sub
    ave_rec = best_all_rec / count_sub
    ave_pre = best_all_pre / count_sub
    ave_f_1 = best_all_f_1 / count_sub
    # 得到所有测试对象的真实标签和预测标签求混淆矩阵，并且将混淆矩阵写入txt
    l = [t for t in range(config['num_class'])]

    ture_all_list = np.array(ture_all_list)
    pred_all_list = np.array(pred_all_list)

    c_m = confusion_matrix(ture_all_list, pred_all_list, labels=l)
    print(c_m)
    print(c_m, file=fp_end)

    end_time = time.time()
    execution_time = end_time - start_time

    print(
        f"The final results of all subjects are shown ave_acc:{ave_acc:.6f},ave_rec:{ave_rec:.6f},ave_pre:{ave_pre:.6f},ave_f_1:{ave_f_1:.6f}\n")
    print(
        f"The final results of all subjects are shown ave_acc:{ave_acc:.6f},ave_rec:{ave_rec:.6f},ave_pre:{ave_pre:.6f},ave_f_1:{ave_f_1:.6f}\n",
        file=fp_end)

    print('Finished the process of train and valid:{}!!'.format(execution_time))
    print('Finished the process of train and valid:{}!!'.format(execution_time), file=fp_end)
    writer.close()
    fp_end.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train_ck_mmew_5_0.1_0.01_0.1.yaml")
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

