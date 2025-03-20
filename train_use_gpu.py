import argparse
import os, glob
import shutil
import warnings
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from models import sam_feat_seg_model_registry
from train_dataset import Dataset


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--epochs', default=50, type=int, required=False,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=30, type=int, required=False,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=4, type=int, required=False,
                    help='mini-batch size ( default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0003, type=float, required=False,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, required=False,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--resume', default='', type=str, required=False,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=7, type=int, required=False,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, required=False,
                    help='GPU id to use.')
parser.add_argument("--iter_2stage", type=int, default=1, required=False, help='第二段迭代次数，训练时固定固定为1')
parser.add_argument("--num_classes", type=int, default=2, required=False,)
parser.add_argument("--save_dir", type=str, default='', required=False,)
parser.add_argument("--load_saved_model", action='store_true', help='whether freeze encoder of the segmenter')
parser.add_argument('--model_type', type=str, default="vit_l", required=False, help='')
parser.add_argument('--format_img', type=str, default='.tif', required=False, help='')
parser.add_argument('--format_img_glob', type=str, default='*.tif', required=False, help='')
parser.add_argument("--img_size", type=int, default=1024, required=False, help='原图尺寸')
parser.add_argument('--img_path', type=str, required=False, default=r'')
parser.add_argument('--label_path', type=str, required=False, default=r'')
parser.add_argument('--model_checkpoint', type=str, required=False, default=r'')
# create model

def iou_score(mask,img_out):
    mask = mask.detach().cpu().numpy()
    img_out = img_out.argmax(dim=1)
    img_out = img_out.detach().cpu().numpy()

    # 计算交并比
    iou = 0
    for i in range(mask.shape[0]):
        temp_mask = mask[i,...]
        temp_img_out = img_out[i,...]
        intersection_0 = ((temp_mask == 0) & (temp_img_out == 0)).sum()
        intersection = (temp_mask == temp_img_out).sum() - intersection_0
        mask_bool = temp_mask > 0
        img_out_bool = temp_img_out > 0
        union = (mask_bool | img_out_bool).sum()
        iou += (intersection+0.05) / (union+0.05)
    iou = iou/mask.shape[0]
    return iou

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = sam_feat_seg_model_registry[args.model_type](num_classes=args.num_classes, checkpoint=args.model_checkpoint, img_size=args.img_size, iter_2stage=args.iter_2stage)

    if args.gpu is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        model.to(device)
    else:
        raise NotImplementedError("No cuda can use.")

    # freeze weights in the image_encoder

    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name:
            param.requires_grad = False
        elif param.requires_grad and "promptcnn_first" in name:
            param.requires_grad = True
        elif param.requires_grad and "seg_decoder_first" in name:
            param.requires_grad = True
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    # optimizer = torch.optim.SGD(params=model.parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_dataset(args)
    # 数据预处理
    train_transform = albu.Compose([
        albu.Resize(args.img_size, args.img_size),
        albu.ShiftScaleRotate(p=0.8),
        albu.OneOf([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Transpose(p=0.5)
        ], p=0.9),
        albu.RandomBrightnessContrast(p=0.8),
        albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ], is_check_shapes=False)

    val_transform = albu.Compose([
        albu.Resize(args.img_size, args.img_size),
        albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ], is_check_shapes=False)
    # 要求img和label的名字一样,修改此处
    img_names = glob.glob(os.path.join(args.img_path, args.format_img_glob)) #数据读取那一块也需要同步修改
    base_names = [os.path.basename(x).split('.')[0] for x in img_names]
    train_names, val_names = train_test_split(base_names, train_size=0.8, random_state=42) #数据集划分

    train_set = Dataset(args=args, names=train_names, transform=train_transform)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_set = Dataset(args=args, names=val_names, transform=val_transform)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    args.save_dir = "./output_experiment/" + args.save_dir
    print(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard' + str(gpu)))


    best_loss = 100

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)

        # train for one epoch
        scheduler.step()
        train(train_loader, model, optimizer, epoch, args, writer)
        val_loss = validate(val_loader, model, optimizer, epoch, args, writer)

        if val_loss < best_loss:
            is_best = True
            best_loss = val_loss
            #修改此处
            filename = 'all_' + 'model_' + str(epoch) + '.pth'
            if is_best:
                print('save model: val_loss = {0}'.format(val_loss))
            save_checkpoint(model.state_dict(), is_best, filename=filename)


def train(train_loader, model, optimizer,  epoch, args, writer):
    train_loss = 0

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # switch to train mode
    model.train()
    # end = time.time()
    for img, label in tqdm(train_loader, total = len(train_loader)):
        img, mask = img.to(device), label.to(device, dtype=torch.long)
        mask = mask.squeeze(dim=1)
        optimizer.zero_grad()
        img_out1, img_out, prompt_embedding = model(img)
        loss = criterion(img_out, mask)*0.3 + criterion(img_out1, mask)*0.7
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    mean_train_loss  = train_loss / len(train_loader)
    print('epoch [{0}]: mean_train_loss={1}'.format(epoch, mean_train_loss))



def validate(val_loader, model, optimizer, epoch, args, writer):
    print('VALIDATE')
    #model.eval()
    model.train()
    iou = 0
    total_num = 0
    val_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    for img, label in tqdm(val_loader, total=len(val_loader)):
        img, label = img.to(device), label.to(device, dtype=torch.long)
        label = label.squeeze(dim=1)
        optimizer.zero_grad()
        img_out1, img_out, prompt_embedding = model(img)
        temp_val_loss = criterion(img_out, label)*0.3 + criterion(img_out1, label)*0.7 #以最后的为主 #0.23
        temp_val_loss.backward()
        optimizer.step()

        temp_val_iou = iou_score(label, img_out)*0.3 + iou_score(label, img_out1)*0.7
        iou += temp_val_iou
        val_loss += temp_val_loss.item()
        total_num += 1
    val_loss = val_loss / total_num
    val_iou = iou / total_num
    print('epoch[{0}]: val_loss={1}'.format(epoch, val_loss))
    print('epoch[{0}]: val_iou={1}'.format(epoch, val_iou))

    return val_loss
    # writer.add_scalar("val_loss", np.mean(loss_list), epoch)
    # return np.mean(loss_list)

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if True:
        torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'all_model_best.pth')


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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    args = parser.parse_args()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)
