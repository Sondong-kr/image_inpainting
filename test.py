import argparse
import torch
from torchvision import transforms

import opt
from coco import Coco
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from torch.utils import data
from tqdm import tqdm

from score import cal_psnr
from score import cal_ssim

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--mask_root', type=str, default='./mask')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=4)# previous default=16 
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_test = Coco(args.root, args.mask_root, img_transform, mask_transform, infer=True)
sampler = data.RandomSampler(dataset_test)
test_loader = data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler)


model = PConvUNet().to(device)
# load_ckpt(args.snapshot, [('model', model)])

model.load_state_dict(torch.load('./model/epoch 11_current_best_model.pt'))
model.eval()
total_psnr = 0.0
total_ssim = 0.0
count = 0
with torch.no_grad():
    for i, (img, mask, gt) in tqdm(enumerate(test_loader)):
        img, mask, gt = img.to(device), mask.to(device), gt.to(device)
        output, _ = model(img, mask)
        output = output.cpu().detach()
        gt = gt.cpu().detach()
        psnr = cal_psnr(output, gt)
        # ssim_value = cal_ssim(output, gt)
        total_psnr += psnr
        # total_ssim += ssim_value
        count += 1

    avg_psnr = total_psnr / count
    # avg_ssim = total_ssim / count/

print('PSNR : {:f}'.format(avg_psnr))
# print('SSIM : {:d}'.format(avg_ssim))

evaluate(model, dataset_test, device, 'result.jpg')
