import os
import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from basicsr.archs.uformer_arch import Uformer
import argparse
from basicsr.archs.unet_arch import U_Net
from basicsr.utils.flare_util import mkdir
from torch.nn import functional as F
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--model_type', type=str, default='Uformer')
parser.add_argument('--model_path', type=str, default='checkpoint/flare7kpp/net_g_last.pth')
parser.add_argument('--output_ch', type=int, default=3)

args = parser.parse_args()
model_type = args.model_type
images_path = os.path.join(args.input, "*.*")
result_path = args.output
pretrain_dir = args.model_path
output_ch = args.output_ch


def load_params(model_path):
    full_model = torch.load(model_path)
    if 'params_ema' in full_model:
        return full_model['params_ema']
    elif 'params' in full_model:
        return full_model['params']
    else:
        return full_model


def demo(images_path, output_path, model_type, output_ch, pretrain_dir):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path = glob.glob(images_path)
    result_path = output_path
    torch.cuda.empty_cache()
    if model_type == 'Uformer':
        model = Uformer(img_size=512, img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type == 'U_Net' or model_type == 'U-Net':
        model = U_Net(img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    else:
        assert False, "This model is not supported!!"
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(512)  # The output should in the shape of 128X
    for i, image_path in tqdm(enumerate(test_path)):
        mkdir(result_path + "deflare/")
        deflare_path = result_path + "deflare/" + image_path.split('/')[-1]

        merge_img = Image.open(image_path).convert("RGB")
        resize2org = transforms.Resize((merge_img.size[1], merge_img.size[0]))
        merge_img_org = to_tensor(merge_img)
        merge_img = resize(merge_img_org)
        merge_img = merge_img.cuda().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            window_size = 32
            _, _, h, w = merge_img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            merge_img_tmp = F.pad(merge_img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            output_img = model(merge_img_tmp)
            _, _, h, w = output_img.size()
            deflare_img = output_img[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]

            deflare_img = merge_img_org.cuda().unsqueeze(0) - resize2org(merge_img - deflare_img)
            torchvision.utils.save_image(deflare_img, deflare_path)


demo(images_path, result_path, model_type, output_ch, pretrain_dir)
