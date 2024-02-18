
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import sys
import time
import csv

from collections import defaultdict
from typing import List
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torchvision
import compressai
from compressai.zoo import load_state_dict
import torch
import os
import math
import torch.nn as nn
from Network import TestModel
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb
from PIL import Image

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, f, outputpath, patch):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/result.csv'
    print('decoding img: {}'.format(f))
########original padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    _, _, height, width = x_padded.size()
    start = time.time()
    out_enc= model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = torch.nn.functional.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = 0
    for s in out_enc["strings"]:
        for j in s:
            if isinstance(j, list):
                for i in j:
                    if isinstance(i, list):
                        for k in i:
                            bpp += len(k)
                    else:
                        bpp += len(i)
            else:
                bpp += len(j)
    bpp *= 8.0 / num_pixels
    # bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    z_bpp = len(out_enc["strings"][1][0])* 8.0 / num_pixels
    y_bpp = bpp - z_bpp

    torchvision.utils.save_image(out_dec["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_dec["x_hat"])
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, y_bpp, z_bpp,
               torch.nn.functional.mse_loss(x, out_dec["x_hat"]).item() * 255 ** 2, psnr(x, out_dec["x_hat"]),
               ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(), enc_time, dec_time, out_enc["time"]['y_enc'] * 1000,
               out_dec["time"]['y_dec'] * 1000, out_enc["time"]['z_enc'] * 1000, out_enc["time"]['z_dec'] * 1000,
               out_enc["time"]['params'] * 1000]
        write = csv.writer(f)

        write.writerow(row)
 
    print("y_size:{}\n".format(out_enc["latent"].shape)) 
    print("y_hat_size:{}\n".format(out_dec["y_hat"].shape))
    print('bpp:{}, PSNR: {}, encoding time: {}, decoding time: {}'.format(bpp, PSNR, enc_time, dec_time))
    

    #visualizaiton
    
    b,c,h,w = out_enc["latent"].shape
    y = out_enc["latent"]
    
    for i in range(c):
        latent_c  = np.abs(np.array(y[0,i,:,:].to('cpu')))
        max_y = latent_c.max()
        min_y = latent_c.min()
        latent_c = (latent_c-min_y)/(max_y-min_y+1e-7)
        #heat_data = pd.DataFrame(latent_c)
        #plt.imsave(os.path.join(outputpath,'map_{}.png'.format(i)), latent_c, vmin=0, vmax=1, cmap='Greys')
        #sns.heatmap(heat_data)
        #plt.savefig(os.path.join(outputpath,'heatmap_{}.png'.format(i)))
        channel_map = latent_c*255
        channel_image = Image.fromarray(channel_map)
        channel_image = channel_image.convert('L')
        channel_image.save(os.path.join(outputpath,'map_{}.png'.format(i)))

    #print(out_enc["latent"][0,3,:,:])
    #print(out_dec["y_hat"][0,3,:,:])

    return {
        "psnr": PSNR,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_entropy_estimation(model, x, f, outputpath, patch):
    x = x.unsqueeze(0) 
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/result.csv'
    print('decoding img: {}'.format(f))
    ########original padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = torch.nn.functional.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    _, _, height, width = x_padded.size()


    start = time.time()
    out_net = model.inference(x_padded)


    elapsed_time = time.time() - start
    out_net["x_hat"] = torch.nn.functional.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    

    y_likelihood = out_net["likelihoods"]["y"]
    channel_num = y_likelihood.shape[1]
    channel_entropy = np.zeros(channel_num)
    for ch_idx in range(channel_num):
        likelihood_ch = y_likelihood[0,ch_idx,:,:]
        channel_entropy[ch_idx] = torch.log(likelihood_ch).sum()/(-math.log(2))
        '''
        likelihood_ch = (torch.log(likelihood_ch))/(-math.log(2))
        likelihood_min = likelihood_ch.min()
        likelihood_max = likelihood_ch.max()
        l_norm = np.array((likelihood_ch-likelihood_min)/(likelihood_max-likelihood_min)*255)
        l_norm  = Image.fromarray(l_norm)
        l_norm = l_norm.convert('L')
        l_norm.save(os.path.join(outputpath,"entropy{}.png".format(ch_idx)))
        '''
    
    x_axis  = np.linspace(0,4,5)
    y_axis  = np.zeros(5)
    group = [16,16,32,64,192]
    index = 0
    ch_now = 0
    for ch_slice in group:
        y_axis[index] = channel_entropy[ch_now:ch_now+ch_slice].sum()
        index+=1
        ch_now+=ch_slice
    
    plt.figure(figsize=(8,4))
    plt.title('entropy of each slice')
    #plt.xticks(np.arange(0,channel_num,10))
    plt.xlabel('channel')
    plt.ylabel('entropy')
    plt.bar(x_axis,y_axis)
    plt.savefig('entropy2.png')

    y_bpp = (torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))

    z_bpp = (torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))

    torchvision.utils.save_image(out_net["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_net["x_hat"])
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp.item() * num_pixels, num_pixels, bpp.item(), y_bpp.item(), z_bpp.item(),
               torch.nn.functional.mse_loss(x, out_net["x_hat"]).item() * 255 ** 2, PSNR,
               ms_ssim(x, out_net["x_hat"], data_range=1.0).item(), elapsed_time / 2.0, elapsed_time / 2.0,
               out_net["time"]['y_enc'] * 1000, out_net["time"]['y_dec'] * 1000, out_net["time"]['z_enc'] * 1000,
               out_net["time"]['z_dec'] * 1000, out_net["time"]['params'] * 1000]
        write = csv.writer(f)
        write.writerow(row)
    return {
        "psnr": PSNR,
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def distortion_without_channel(model, x, f, outputpath, patch):
    x = x.unsqueeze(0) 
    imgpath = f.split('/')
    imgpath[-2] = outputpath
    imgPath = '/'.join(imgpath)
    csvfile = '/'.join(imgpath[:-1]) + '/result.csv'
    print('decoding img: {}'.format(f))
    ########original padding
    channel_num,h, w = x.size(1),x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = torch.nn.functional.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    _, _, height, width = x_padded.size()
    start = time.time()
    #out_net = model.mask_inference2(x_padded)
    #np.save('psnr2.npy',out_net)
    out_net = np.load('psnr2.npy')
    x_axis  = np.linspace(0,4,5,dtype=int)
    y_axis  = out_net 
    plt.figure(figsize=(8,4))
    plt.title('psnr without each slice')
    plt.xlabel('channel')
    plt.ylabel('psnr')
    plt.bar(x_axis,y_axis)
    plt.savefig('psnr2.png')


   


def eval_model(model, filepaths, entropy_estimation=False, distortion_estimation=False, half=False, outputpath='Recon', patch=576):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    imgdir = filepaths[0].split('/')
    imgdir[-2] = outputpath
    imgDir = '/'.join(imgdir[:-1])
    if not os.path.isdir(imgDir):
        os.makedirs(imgDir)
    csvfile = imgDir + '/result.csv'
    if os.path.isfile(csvfile):
        os.remove(csvfile)
    with open(csvfile, 'w') as f:
        row = ['name', 'bits', 'pixels', 'bpp', 'y_bpp', 'z_bpp', 'mse', 'psnr(dB)', 'ms-ssim', 'enc_time(s)', 'dec_time(s)', 'y_enc(ms)',
               'y_dec(ms)', 'z_enc(ms)', 'z_dec(ms)', 'param(ms)']
        write = csv.writer(f)
        write.writerow(row)
    for f in filepaths:
        x = read_image(f).to(device)
        if entropy_estimation:
            rv = inference_entropy_estimation(model, x, f, outputpath, patch) 
        elif distortion_estimation:
            rv  = distortion_without_channel(model,x,f,outputpath,patch)
        else:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, f, outputpath, patch)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

def setup_args():
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument(
        "--output_path",
        help="result output path",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "--distortion-estimation",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        required=True,
        help="checkpoint path",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=256,
        help="padding patch size (default: %(default)s)",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(['--dataset','/mnt/data1/jingwengu/kodak1','--output_path','./ELIC_0004_ENTROPY','--entropy-estimation','-p',
                              '/mnt/data1/jingwengu/pretrained_models/ELIC/ELIC_0004_ft_3980_Plateau.pth.tar','--patch','64'])

    filepaths = collect_images(args.dataset)
    filepaths = sorted(filepaths)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    state_dict = load_state_dict(torch.load(args.paths))
    model_cls = TestModel()
    model = model_cls.from_state_dict(state_dict).eval()

    results = defaultdict(list)

    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")

    metrics = eval_model(model, filepaths, args.entropy_estimation, args.distortion_estimation, args.half, args.output_path, args.patch)
    for k, v in metrics.items():
        results[k].append(v)

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    
    sys.argv.append("python")
    sys.argv.append("Inference.py")
    sys.argv.append("--dataset")
    sys.argv.append("/mnt/data1/jingwengu/kodak1")
    sys.argv.append("--output_path")
    sys.argv.append("./ELIC_0004/ELIC_0004_rate")
    sys.argv.append("-p ")
    sys.argv.append("/mnt/data1/jingwengu/pretrained_models/ELIC/ELIC_0004_ft_3980_Plateau.pth.tar")
    sys.argv.append("--patch") 
    sys.argv.append("64") 
    
    main(sys.argv[1:])
