import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
def cal_psnr(image1, image2):

    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    # Calculate mse
    mse = np.mean((image1 - image2) ** 2) 

    if image1.dtype == 'np.float32':
        max_i = 255.0
    else:
        max_i = 1.0
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_i / np.sqrt(mse))
    return psnr

def cal_ssim(image1, image2):
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()

    ssim_val, diff = ssim(image1, image2, win_size= 7, channel_axis= 0, full=True)
    diff = (diff * 255).astype("uint8")
    return ssim_val