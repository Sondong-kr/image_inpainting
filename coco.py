import random
import torch
from PIL import Image
from glob import glob


class Coco(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform, data_num = 0, infer = False):
        super(Coco, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.infer = infer
        self.data_num = data_num

        # use about 8M images in the challenge dataset
        if self.infer:
            self.paths = glob('{:s}/test2017/*.jpg'.format(img_root))
            self.mask_paths = glob('{:s}/test_mask/*.png'.format(mask_root))
        else:
            self.paths = glob('{:s}/train2017/*'.format(img_root))[:self.data_num]
            self.mask_paths = glob('{:s}/40_69_mask/*.png'.format(mask_root))
            # self.mask_paths = glob('{:s}/test_mask/*.png'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
