import numpy as np
from skimage.metrics import structural_similarity as ssim
def cal_psnr(image1, image2):
    # Calculate mse
    mse = np.mean((image1 - image2) ** 2) 

    if image1.dtype == 'torch.float32':
        max_i = 255.0
    else:
        max_i = 1.0
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_i / np.sqrt(mse))
    return psnr

def cal_ssim(image1, image2):
    ssim_val, diff = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    return ssim_val