import importlib
import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
from attack import get_parser

module = importlib.import_module('.data_util', package='util')
DataUtils = getattr(module, 'DataUtils')

use_cuda=True
image_nc=3
epochs = 60
batch_size = 64
BOX_MIN = 0
BOX_MAX = 1


args = get_parser()

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = r"weights\Final_LPRNet_model.pth"
targeted_model = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = len(CHARS)

img_dirs = os.path.expanduser(args.attack_img_dirs)
dataset = LPRDataLoader(img_dirs.split(','), args.img_size,
                             args.lpr_max_len)
dataloader = torch.utils.data.DataLoader(dataset,
                                    batch_size,
                                    shuffle=True,
                                    collate_fn=DataUtils.collate_fn)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)



advGAN.train(dataloader, epochs)

