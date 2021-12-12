import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from BLOCKS import baseline

# set here
DATASET = 0        #    0: FER2013 1: RAFD 2: SFEW
NORMALIZE = 0       #   0: FER2013 1: RAFD 2: SFEW
BATCH = 128

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth.tar')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/'+time_str+'model_best.pth.tar')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', dest='lr')
parser.add_argument('--factor', default=0.1, type=float, metavar='FT')
parser.add_argument('--af', '--adjust-freq', default=30, type=int, metavar='N', help='adjust learning rate frequency')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--gpu', type=str)
args = parser.parse_args()

if DATASET == 0:
    args.data = '/home/user1/datasets/fer2013/'
elif DATASET == 1:
    args.data = '/home/user1/datasets/RAFD/'
else:    
    args.data = '/home/user1/datasets/changed_SFEW/'


args.batch_size = BATCH

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  
    best_acc = 0

    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    model_cla = baseline.baseline()
    model_cla = torch.nn.DataParallel(model_cla).cuda() 
    model_cla.module.fc = nn.Linear(1024, 7).cuda()  
    criterion_val = nn.CrossEntropyLoss().cuda()
    criterion_train = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_cla.parameters(),             
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    recorder = RecorderMeter(args.epochs)
    cudnn.benchmark = True
    traindir = os.path.join(args.data, 'train')             
    valdir = os.path.join(args.data, 'test')                


    if NORMALIZE == 0:
        normalize = transforms.Normalize(mean=[0.50383407,0.50383407,0.50383407], 
                                      std=[0.24635993,0.24635993,0.24635993])                                      
    elif NORMALIZE == 1:
        normalize = transforms.Normalize(mean=[0.72236615,0.63140035,0.5914867],  
                                      std=[0.20470998,0.23333736,0.25737643])
    elif NORMALIZE == 2:
        normalize = transforms.Normalize(mean=[0.33443722,0.224762,0.17571008], 
                                      std=[0.18094133,0.13987286,0.13730428])                                          
    train_dataset = datasets.ImageFolder(traindir,                       
                                         transforms.Compose([     
                                                             transforms.RandomResizedCrop((224, 224)), #
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),  
                                                             normalize]))
    test_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([
                                                            transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset,                  
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,                      
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model_cla, criterion_val, args)   
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args) 
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:                                          
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')


        train_acc, train_los = train(train_loader, model_cla, criterion_train, optimizer, epoch, args)
        val_acc, val_los = validate(val_loader, model_cla, criterion_val, args)

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'log.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'state_dict': model_cla.state_dict()}, is_best, args)

        
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')


def train(train_loader, model_cla, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    soft_max = nn.Softmax(dim=1)

    # switch mode
    model_cla.train()

    for i, (images, target) in enumerate(train_loader):

        # length = len(images)
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model_cla(images)
        output_prob = soft_max(output)

        loss = criterion(output_prob,target)                   

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')


    model.eval()

    with torch.no_grad():           
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()


            output = model(images)
            loss = criterion(output, target)


            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))       
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' *** Accuracy {top1.avg:.3f}  *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


def cross_entropy(predict_label, true_label):
    return torch.mean(- true_label * torch.log(predict_label))


class AverageMeter(object):                         
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)           
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (args.factor ** (epoch // args.af))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))            
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True) 
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
