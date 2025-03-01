from __future__ import print_function
import os
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchattacks
from model.LPRNet import build_lprnet
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from matplotlib.font_manager import FontProperties
from attack import get_parser
import importlib
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from pymoo.algorithms.moo.nsga2 import NSGA2  
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling,IntegerRandomSampling,BinaryRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

module = importlib.import_module('.data_util', package='util')
DataUtils = getattr(module, 'DataUtils')

args = get_parser()

# 攻击目标模型
pretrained_model = args.attack_model

# 定义我们正在使用的设备
print("CUDA Available: ", torch.cuda.is_available())

# 初始化网络
lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
device = torch.device("cuda:0" if args.cuda else "cpu")
lprnet.to(device)
print("Successful to build network!")

# 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
if pretrained_model:
    lprnet.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    print("load pretrained model successful!")
else:
    print("[Error] Can't found pretrained mode, please check!")

# 在评估模式下设置模型。在这种情况下，这适用于Dropout图层
lprnet.eval()

batch_size = args.batch_size
img_dirs = os.path.expanduser(args.attack_img_dirs)
test_dataset = LPRDataLoader(img_dirs.split(','), args.img_size,
                             args.lpr_max_len)
epoch_size = len(test_dataset)
batch_iterator = iter(torch.utils.data.DataLoader(test_dataset,
                                    batch_size,
                                    shuffle=True,
                                    collate_fn=DataUtils.collate_fn))


class AdversarialAttack(Problem):
    def __init__(self, model, x_orig, y_orig, p_norm=2, c=0.1):
        super().__init__(n_var=1,n_obj=2)
        self.model = model
        self.x_orig = x_orig
        self.y_orig = y_orig
        self.p_norm = p_norm
        self.c = c

    def evaluate(self, out, *args, **kwargs):
        modifier=Variable(torch.zeros_like(self.x_orig).to(device).float())
        x_adv = modifier+self.x_orig
        x_adv = torch.clamp(x_adv, 0, 1)  # 确保对抗样本在[0, 1]范围内
        

        # 计算扰动范数损失
        distortion_loss=torch.dist(x_adv, self.x_orig, p=self.p_norm)
        # 计算对抗性损失
        logits = self.model(x_adv)
        ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        x = self.y_orig.numel()
        target_l = (x, )

        loss = ctc_loss(log_probs,
                        self.y_orig,
                        input_lengths=(18, ),
                        target_lengths=target_l)

        model.zero_grad()
        loss.backward()
        adv_loss = -loss
        out={}
        # 返回两个目标函数值
        out["F"] = [distortion_loss, adv_loss]


# 循环遍历测试集中的所有示例
data, target, lengths = next(batch_iterator)
# 把数据和标签发送到设备
data, target = data.to(device), target.to(device)
img = data
img = img.cpu().numpy()

model=lprnet

problem = AdversarialAttack(model, data, target)
algorithm = NSGA2(
   pop_size=90, # z种群数量
   n_offsprings=100, # 每代的数量
   sampling= FloatRandomSampling(), #抽样设置
    #交叉配对
   crossover=SBX(prob=0.9 #交叉配对概率
                 , eta=15), #配对效率
   #变异
   mutation=PM(prob=0.8 #编译概率
               ,eta=20),# 配对效率
   eliminate_duplicates=True
)

res = minimize(problem, algorithm, termination=('n_gen', 100), verbose=False)
print()