import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from evaluation import evaluate
from score import cal_psnr
from score import cal_ssim
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from coco import Coco
from util.io import load_ckpt
from util.io import save_ckpt


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--mask_root', type=str, default='./mask')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--train_data_num', type=int, default=40000) #  default=118287
parser.add_argument('--max_iter', type=int, default=200000) #  default=1000000
parser.add_argument('--batch_size', type=int, default=4) # previous default=16 
parser.add_argument('--n_threads', type=int, default=0) # previous default = 16
parser.add_argument('--save_model_interval', type=int, default=7500) #default=50000
parser.add_argument('--vis_interval', type=int, default=50000)
parser.add_argument('--log_interval', type=int, default=10000)
parser.add_argument('--image_size', type=int, default=256) # default=256
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_train = Coco(args.root, args.mask_root, img_tf, mask_tf, data_num = args.train_data_num, infer= False)
dataset_val = Coco(args.root, args.mask_root, img_tf, mask_tf, infer= True)

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))
print(len(dataset_train))
model = PConvUNet().to(device)
model.load_state_dict(torch.load('./model/epoch 16_current_best_model.pt'))

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

total_psnr = 0
total_ssim = 0
min_loss = float('inf')
for i in tqdm(range(start_iter, args.max_iter)):
    model.train()

    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    epoch_size = args.train_data_num/args.batch_size
    epoch = int(i//epoch_size)
    
    # psnr = cal_psnr(output, gt)
    # ssim = cal_ssim(output, gt)
    # total_psnr += psnr
    # total_ssim += ssim
    # avg_psnr = total_psnr / len(epoch_size)
    # avg_ssim = total_ssim / len(epoch_size)

    if (i % epoch_size==0) and i>0:
        print('loss for {:d}epoch: {:f}'.format(epoch, loss))
        # print('PSNR for {:d}epoch: {:d}'.format(epoch, avg_psnr))
        # print('SSIM for {:d}epoch: {:d}'.format(epoch, avg_ssim))
        if min_loss > loss:
            min_loss = loss
            best_model = model
            state_dict = best_model.state_dict()
            best_epoch = (i % epoch_size) +1
            print(f"\nEpoch [{best_epoch}] New Minimum Valid Loss!")
            print("..save current best model..")
            model_name = f'epoch {epoch}_current_best_model.pt'
            torch.save(state_dict, './model'+'/'+model_name)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        evaluate(model, dataset_val, device,
                 '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))

writer.close()
