import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    img1_tensor = img1_tensor * 2.0 - 1.0
    img2_tensor = img2_tensor * 2.0 - 1.0
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0, 0, 0, 0]


def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    gt_folder = args['gt'] + '/*'
    input_folder = args['input'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0
    for i in tqdm(range(n)):
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)
    ssim /= n
    psnr /= n
    lpips_val /= n
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--gt', type=str, default=None)
    args = vars(parser.parse_args())
    calculate_metrics(args)
