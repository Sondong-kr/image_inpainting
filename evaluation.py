import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from util.image import unnormalize
import random

def evaluate(model, dataset, device, filename):
    random_indices = random.sample(range(40000), 8)
    image, mask, gt = zip(*[dataset[i] for i in random_indices])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)
