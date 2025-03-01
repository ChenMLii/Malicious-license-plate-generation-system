import os
from PIL import Image
import numpy as np
import torch
import importlib
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.LPRNet import build_lprnet
from attack import get_parser
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
import models

module = importlib.import_module('.data_util', package='util')
DataUtils = getattr(module, 'DataUtils')

use_cuda=True
image_nc=3
batch_size = 1

gen_input_nc = image_nc
args = get_parser()

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = r"weights\Final_LPRNet_model.pth"
targeted_model = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
targeted_model = targeted_model.to(device)

# load the generator of adversarial examples
pretrained_generator_path = r'weights\netG_epoch_60.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

img_dirs = os.path.expanduser(args.attack_img_dirs)
dataset = LPRDataLoader(img_dirs.split(','), args.img_size,
                             args.lpr_max_len)
dataloader = torch.utils.data.DataLoader(dataset,
                                    batch_size,
                                    shuffle=True,
                                    collate_fn=DataUtils.collate_fn)

#标签转换
def trans_label(label):
    lb = ""
    for i in label:
        lb += CHARS[i]
    return lb

for i, data in enumerate(dataloader, start=0):
    test_img, test_label, lengths = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)

    output = targeted_model(adv_img)
    # 保存生成图像

    final_pred = DataUtils.greedy_decoder(output)
    final_pred = torch.tensor(final_pred).to(device)
    image_array = adv_img.squeeze().detach().cpu().numpy()
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array.transpose(1, 2, 0))
    image = image.convert('RGB')
    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))
    lb1 = test_label.tolist()
    lbx = [int(x) for x in lb1]
    lb2 = trans_label(lbx)
    file_name = lb2 + '.jpg'
    folder_path = r'./AdvGan Adversarial examples'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    image.save(file_path)