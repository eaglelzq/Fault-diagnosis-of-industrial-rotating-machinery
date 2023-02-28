# -*- coding:utf-8-*-
"""
作者:Eagle
日期:2022年08月02日
"""
from __future__ import print_function
import argparse
import torch
from myDataset import Train_Dataset,Test_Dataset,Target_Dataset
from mmd import MMDLoss
from torch.utils import data as data_
from DAN import DANNet
import numpy as np

# Training settings
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DANN')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_epoch', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=233, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--l2_decay', type=float, default=5e-4,
                        help='the L2  weight decay')
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--train_path', type=str, default=r'D:\Study\competition_lerning\Yidongbei2\Data_version\training\total_train.csv',
                        help='Step3中生成的保存训练集的csv文件地址')
    parser.add_argument('--target_path', type=str, default=r'D:\Study\competition_lerning\Yidongbei2\Data_version\training\img_unlabeled',
                        help='Step2中生成的无标签的训练集的图片保存文件夹')
    parser.add_argument('--diff_lr', type=bool, default=True,
                        help='the fc layer and the sharenet have different or same learning rate')
    parser.add_argument('--gamma', type=int, default=1,
                        help='the fc layer and the sharenet have different or same learning rate')
    parser.add_argument('--num_class', default=8, type=int,
                        help='the number of classes')
    parser.add_argument('--epoch_based_training', type=bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=50, help="Used in Iteration-based training")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    return parser

def load_data(args):
    Trainset = Train_Dataset(args.train_path)
    source_train_loader = data_.DataLoader(Trainset,batch_size=args.batch_size,shuffle=True)
    targetset = Target_Dataset(args.target_path)
    target_train_loader = data_.DataLoader(targetset , batch_size=args.batch_size, shuffle=True)
    return source_train_loader,target_train_loader

def _test(model, target_test_loader):
    model.eval()
    test_loss = AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            # data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg
def get_optimizer(model,args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = [{'params':model.sharedNet.parameters(),'lr':0.1*initial_lr},
              {'params':model.bottleneck.parameters(),'lr':0.1*initial_lr},
              {'params':model.domain_classifier.parameters(),'lr':0.1*initial_lr},
              {'params': model.dcis.parameters(),'lr':0.1*initial_lr},
              {'params':model.source_fc.parameters(),'lr':1.0*initial_lr}]
    optimizer = torch.optim.SGD(params,lr=args.lr,momentum=args.momentum,weight_decay=args.l2_decay)
    return optimizer
def get_scheduler(optimizer,args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x:args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler
def train(source_loader,target_loader,model,optimizer,lr_scheduler,args):
    len_soure_loader = len(source_loader)
    len_target_loader = len(target_loader)
    n_batch = min(len_target_loader,len_soure_loader)
    iter_source,iter_target = iter(source_loader),iter(target_loader)
    best_acc = 95.3
    stop = 0
    log = []
    # class_criterion = torch.nn.CrossEntropyLoss()
    # domain_clf_criterion = torch.nn.CrossEntropyLoss()
    # domain_criterion = torch.nn.CrossEntropyLoss()
    class_criterion = torch.nn.NLLLoss().to('cuda')
    domain_clf_criterion = torch.nn.NLLLoss().to('cuda')
    domain_criterion = torch.nn.NLLLoss().to('cuda')
    for e in range(1,args.n_epoch+1):
        model.train()
        train_loss_clf = AverageMeter()
        train_loss_domain = AverageMeter()
        train_loss_total = AverageMeter()
        if max(len_target_loader,len_soure_loader)!=0:
            iter_source, iter_target = iter(source_loader), iter(target_loader)
        train_correct = 0
        domain_correct = 0

        global D_M, D_C, MU
        d_m = 0
        d_c = 0
        if D_M == 0 and D_C == 0 and MU == 0:
            MU = 0.5
        else:
            D_M = D_M / n_batch
            D_C = D_C / n_batch
            MU = 1 - D_M / (D_M + D_C)#动态计算神经网络学习边缘分布和条件分布的动态重要型
        for i in range(n_batch):
            p = float(i + e * n_batch) / (args.n_epoch) /n_batch#进程相对值，表示当前迭代次数相对于总迭代次数的比率
            # alpha逐渐增大，alpha=0时领域分类损失不会回传到编码器网络中，只有领域分类器得到训练；随着训练的进行， alpha 逐渐增加，编码器得到训练，并开始逐步生成可以混淆领域分类器的特征。
            alpha = 2./(1.+np.exp(-10*p))-1#
            data_source,source_label = next(iter_source)
            data_target = next(iter_target)
            tmpd_c = 0
            #计算源域损失
            """
            source_task_clf:原分类任务的结果
            source_domain_clf:目标域 和源域的判别结果
            s_domain_output:每个类判别器输出的分类概率
            s_feat2:特征提取层输出的特征
            """
            source_task_clf,source_domain_clf,s_domain_output,Sfeat2 = model(data_source,alpha)
            err_task_source = class_criterion(source_task_clf,source_label)
            err_domain_clf_source = domain_clf_criterion(source_domain_clf,
                                           torch.zeros(source_domain_clf.shape[0],dtype=torch.long).to('cuda'))
            err_domain_source = domain_criterion(s_domain_output,torch.zeros(s_domain_output.shape[0],dtype=torch.long).to('cuda'))
            #计算目标域损失
            taget_label, target_domain_clf,t_domain_output,Tfeat2 = model(data_target, alpha)
            err_domain_clf_target = domain_clf_criterion(target_domain_clf,torch.ones(target_domain_clf.shape[0],dtype=torch.long).to('cuda'))
            err_domain_target = domain_criterion(t_domain_output,torch.ones(t_domain_output.shape[0],dtype=torch.long).to('cuda'))


            local_loss= 0.01*(err_domain_target + err_domain_source)#
            gloable_loss  =0.05*( err_domain_clf_target+err_domain_clf_source)
            train_pred = torch.max(source_task_clf, 1)[1]
            train_correct += torch.sum(train_pred == source_label)
            domain_pred = torch.max(target_domain_clf, 1)[1]
            domain_correct += torch.sum(domain_pred ==torch.ones(t_domain_output.shape[0],dtype=torch.long).to('cuda'))
            lossMMD2 = MMDLoss()
            mloss = 3*lossMMD2(Sfeat2,Tfeat2)
            tmpd_c += 2 * (1 - 2 * (local_loss))
            tmpd_c/=args.num_class
            d_c = d_c + tmpd_c.cpu().item()
            d_m = d_m + 2 * (1 - 2 * gloable_loss.cpu().item())
            join_loss = (1 - MU) * gloable_loss + MU * local_loss
            loss = err_task_source+(5*gloable_loss + 5*local_loss)
            # loss = err_task_source + 5*join_loss+mloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            train_loss_clf.update(err_task_source.item())
            train_loss_domain.update(gloable_loss.item()+local_loss.item())
            train_loss_total.update(loss.item())
        D_M = np.copy(d_m).item()
        D_C = np.copy(d_c).item()
        log.append([train_loss_clf.avg,train_loss_domain.avg,train_loss_total.avg])

        train_acc = 100. * train_correct / n_batch/32
        domain_acc = 100. * domain_correct / n_batch/32
        print("MU:",MU)
        info = 'Epoch:[{:2d}/{}],cls_loss:{:.4f},train_acc:{:.4f},transfer_loss:{:.4f},domain_acc:{:.4f},MMDloss:{:.4f},total_loss:{:.4f}'.format(
            e,args.n_epoch,train_loss_clf.avg,train_acc,train_loss_domain.avg,domain_acc,mloss,train_loss_total.avg
        )
        stop += 1
        print(info)
    torch.save(model.state_dict(), 'model_saver/my_model.params')
def main():
    parser = get_parser()
    args = parser.parse_args()

    # set_random_seed(args.seed)
    source_loader, target_train_loader = load_data(args)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = DANNet()
    model.to('cuda')
    # model.load_state_dict(torch.load('./model_saver/my_model.params'))
    optimizer = get_optimizer(model, args)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, model, optimizer, scheduler, args)
if __name__ == '__main__':
    D_M = 0
    D_C = 0
    MU = 0
    main()
