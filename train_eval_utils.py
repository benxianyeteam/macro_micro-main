import sys
from torchmetrics import MetricCollection, Accuracy, F1Score, Recall, Precision
from tqdm import tqdm
import torch
import utils
from utils import *
from utils.distributed_utils import reduce_value, is_main_process, warmup_lr_scheduler
from models.utils_functions import initialize_nodes_edges, OT
# config=config,
#                             feature_encoder=feature_encoder,
#                             IDE_block=IDE_block,
#                             feature_encoder_optim=feature_encoder_optim,
#                             IDE_block_optim=IDE_block_optim,
#                             cross_loss=cross_loss,
#                             mmd_loss=mmd_loss,
#                             train_macro_loader=train_macro_loader,
#                             train_micro_loader=train_micro_loader,
#                             epoch=epoch,
#                             warmup=True
def train_one_epoch(config, feature_encoder, IDE_block, feature_encoder_optim, IDE_block_optim, cross_loss, mmd_loss, train_macro_loader, train_micro_loader, epoch, use_amp=False, warmup=True):
    feature_encoder.train()
    IDE_block.train()
    feature_encoder_optim.zero_grad()
    IDE_block_optim.zero_grad()
    # 设置评价指标
    metric_collection = MetricCollection(
        {
            'ACC': Accuracy(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="weighted"),
            # ACC
            'REC': Recall(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="macro"),  # REC
            'PRE': Precision(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="macro"),
            # PRE
            'F_1': F1Score(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="macro")  # F_1
        }
    )
    '''# 获取宏表情和微表情加载器的迭代器'''
    train_macro_iter = iter(train_macro_loader)
    train_micro_iter = iter(train_micro_loader)
    # 比较一下加载器的长度，按照长的来进行迭代
    num_iterations = len(train_macro_loader) if (len(train_macro_loader) > len(train_micro_loader)) else len(
        train_micro_loader)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, (num_iterations - 1))

        lr_scheduler = warmup_lr_scheduler(feature_encoder_optim, warmup_iters, warmup_factor)
    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(range(num_iterations), file=sys.stdout)

    enable_amp = use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)


    # 用来统计各项loss
    train_clf_loss = utils.AverageMeter()
    train_cross_att_loss = utils.AverageMeter()
    train_domain_loss = utils.AverageMeter()
    train_total_loss = utils.AverageMeter()
    total_count_train = 0
    # 正确率
    correct_train = 0.0

    for batch_index, _ in enumerate(data_loader):
        # try:  # 宏表情
        #     data_macro_on, data_macro_apex, label_macro, domain_label_macro, landmark_macro = next(
        #         train_macro_iter)
        # except:
        #     train_macro_iter = iter(train_macro_loader)
        #     data_macro_on, data_macro_apex, label_macro, domain_label_macro, landmark_macro = next(
        #         train_macro_iter)
        # try:  # 微表情
        #     data_micro_on, data_micro_apex, label_micro, domain_label_micro, landmark_micro = next(
        #         train_micro_iter)
        # except:
        #     train_micro_iter = iter(train_micro_loader)
        #     data_micro_on, data_micro_apex, label_micro, domain_label_micro, landmark_micro = next(
        #         train_micro_iter)
        # 继续处理数据...
        data_macro_on, data_macro_apex, label_macro, domain_label_macro, landmark_macro = get_next_batch(
            train_macro_iter, train_macro_loader)
        data_micro_on, data_micro_apex, label_micro, domain_label_micro, landmark_micro = get_next_batch(
            train_micro_iter, train_micro_loader)

        # 将宏表情放到cuda
        data_macro_on, data_macro_apex = data_macro_on.cuda(), data_macro_apex.cuda()
        label_macro = label_macro.cuda()
        domain_label_macro = domain_label_macro.cuda()
        landmark_macro = landmark_macro.cuda()
        # 将微表情放到cuda
        data_micro_on, data_micro_apex = data_micro_on.cuda(), data_micro_apex.cuda()
        label_micro = label_micro.cuda()
        domain_label_micro = domain_label_micro.cuda()
        landmark_micro = landmark_micro.cuda()

        # 使用广播机制和条件索引来填充矩阵
        transfer_matrix = (label_macro.unsqueeze(1) == label_micro.unsqueeze(0)).float()
        # 前向传播
        with torch.cuda.amp.autocast(enabled=enable_amp):
            data_on = torch.cat((data_macro_on, data_micro_on), dim=0)
            data_apex = torch.cat((data_macro_apex, data_micro_apex), dim=0)
            label = torch.cat((label_macro, label_micro), dim=0)
            landmark = torch.cat((landmark_macro, landmark_micro), dim=0)
            feature, clf = feature_encoder((data_apex - data_on), landmark, transfer_matrix, True)

            # Compute Feature
            feature_macro = feature.narrow(0, 0, feature.size(0) // 2)  # Batch/2 * (64)
            feature_micro = feature.narrow(0, feature.size(0) // 2, feature.size(0) // 2)  # Batch/2 * (64+320)

            label_one_hot_macro, distribution_graph_macro, feature_graph_macro = initialize_nodes_edges(
                feature_macro,
                label_macro,
                config['num_class'])
            label_one_hot_micro, distribution_graph_micro, feature_graph_micro = initialize_nodes_edges(
                feature_micro,
                label_micro,
                config['num_class'])

            # Compute Feature 域内迁移
            point_nodes_macro, point_similarities_macro, distribution_nodes_macro, distribution_similarities_macro = IDE_block(
                feature_macro, feature_graph_macro, label_one_hot_macro, distribution_graph_macro)
            point_nodes_micro, point_similarities_micro, distribution_nodes_micro, distribution_similarities_micro = IDE_block(
                feature_micro, feature_graph_micro, label_one_hot_micro, distribution_graph_micro)
            # 域间迁移
            '''cross attention'''
            cross_att_loss = mmd_loss(point_nodes_macro, point_nodes_micro)

            '''domain alignment'''
            ot_ep, wd_ep, gwd_ep = OT(point_nodes_macro, point_nodes_micro)
            ot_ed, wd_ed, gwd_ed = OT(distribution_nodes_macro, distribution_nodes_micro)
            domain_loss = ot_ep + ot_ed

            '''cross entropy'''
            clf_loss = cross_loss(clf, label)

            loss = clf_loss + 1 * domain_loss + 0.01 * cross_att_loss  # 0.01

            # 统计损失
            clf_loss = reduce_value(clf_loss, average=True)
            train_clf_loss.update(clf_loss.item()) # 交叉熵损失

            cross_att_loss = reduce_value(cross_att_loss, average=True)
            train_cross_att_loss.update(cross_att_loss.item()) # mmd损失

            domain_loss = reduce_value(domain_loss, average=True)
            train_domain_loss.update(domain_loss.item())# ot 损失

            loss = reduce_value(loss, average=True)
            train_total_loss.update(loss.item()) # 总的损失
            '''
                            采用三种方式求解准确率
                            '''
            pred = torch.argmax(clf, dim=1)
            # 打印真实标签和预测标签
            # print(f'The true labels:{label}')
            # print(f'The pred labels:{pred}')
            # 将训练结果写入txt
            # print(f'The true labels:{label}',file=fp)
            # print(f'The pred labels:{pred}',file=fp)

            # 累计正确的值
            correct_train += (pred == label).sum()
            total_count_train += label.size(0)
            # print(f'Statistic of batch :{(pred == label).sum() / label.size(0)}')
            # 采用自己的定义求解准确率
            metric_collection(pred.detach().cpu(), label.detach().cpu())  # 更新度量状态并返回当前批次上计算的度量结果 在batch循环内
            # 所有batch的我的acc
        # ===================backward=====================
        scaler.scale(loss).backward()

        scaler.step(feature_encoder_optim)
        scaler.step(IDE_block_optim)

        feature_encoder_optim.zero_grad()
        IDE_block_optim.zero_grad()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                train_total_loss.sum / (batch_index + 1),
                correct_train / total_count_train,
                feature_encoder_optim.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()
    train_metrics = metric_collection.compute()  # 返回在所有批次上计算的最终结果
    # print(f"Metrics on all data: {train_metrics}")
    # print(f"Metrics on all data: {train_metrics}", file=fp)
    metric_collection.reset()  # 重置状态，以便为下一个验证阶段做好准备。
    acc_train = correct_train / total_count_train
    print(f'Statistic of all batch in train process :{acc_train}')

    # 等待所有进程计算完毕
    torch.cuda.synchronize()

    return train_total_loss.sum, train_clf_loss.sum, train_cross_att_loss.sum, train_domain_loss.sum, train_metrics

# config=config,
#                      feature_encoder=feature_encoder,
#                      cross_loss=cross_loss,
#                      valid_micro_loader=valid_micro_loader,
def evaluate(config, feature_encoder, cross_loss, valid_micro_loader):
    feature_encoder.eval()
    # 设置评价指标
    metric_collection = MetricCollection(
        {
            'ACC': Accuracy(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="weighted"),
            # ACC
            'REC': Recall(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="macro"),  # REC
            'PRE': Precision(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="macro"),
            # PRE
            'F_1': F1Score(task="multiclass", num_classes=config['num_class'], threshold=1. / 5, average="macro")  # F_1
        }
    )

    # 验证集样本个数
    num_samples = len(valid_micro_loader.dataset)

    # 用来统计各项loss
    valid_clf_loss = utils.AverageMeter()

    # 用于存储预测正确的样本个数
    sum_num = 0

    # 在进程0中打印验证进度
    if is_main_process():
        valid_micro_loader = tqdm(valid_micro_loader, file=sys.stdout)
    else:
        pass
    # 正确率
    correct_valid = 0.0
    total_count_valid = 0
    with torch.no_grad():
        label_all = []
        pred_all = []
        for batch_index, (input_on, input_apex, label, domain, landmark) in enumerate(valid_micro_loader):
            # 将数据放到cuda上
            input_on, input_apex = input_on.cuda(), input_apex.cuda()
            label = label.cuda()
            domain = domain.cuda()
            landmark = landmark.cuda()
            features, clf = feature_encoder((input_apex - input_on), landmark)

            clf_loss = cross_loss(clf, label)
            # 统计损失
            clf_loss = reduce_value(clf_loss, average=True)
            valid_clf_loss.update(clf_loss.item())
            '''
                                       采用三种方式求解准确率
                                       '''
            pred = torch.argmax(clf, dim=1)
            label_all.extend(label.detach().cpu().numpy())
            pred_all.extend(pred.detach().cpu().numpy())
            # 打印真实标签和预测标签
            # print(f'The true labels:{label}')
            # print(f'The pred labels:{pred}')
            # # 将训练结果写入txt
            # print(f'The true labels:{label}',file=fp)
            # print(f'The pred labels:{pred}',file=fp)

            # 累计正确的值
            correct_valid += (pred == label).sum()
            total_count_valid += label.size(0)

            # 采用自己的定义求解准确率
            metric_collection(pred.detach().cpu(), label.detach().cpu())  # 更新度量状态并返回当前批次上计算的度量结果
        valid_metrics = metric_collection.compute()  # 返回在所有批次上计算的最终结果
        # print(f"Metrics on all data: {val_metrics}")
        # print(f"Metrics on all data: {val_metrics}", file=fp)
        metric_collection.reset()  # 重置状态，以便为下一个验证阶段做好准备。

    # 等待所有进程计算完毕
    torch.cuda.synchronize()

    correct_valid = reduce_value(correct_valid, average=False)
    acc_valid = correct_valid / total_count_valid
    print(f'Statistic of all batch in valid process :{acc_valid}')

    return valid_clf_loss.sum, valid_metrics, label_all, pred_all






