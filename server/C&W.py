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

module = importlib.import_module('.data_util', package='util')
DataUtils = getattr(module, 'DataUtils')

args = get_parser()
# 设置不同扰动大小
#epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
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

def c_w_loss(data, target_labels, model, confidence=1):
    """
    目标函数,用于生成对抗性样本
    x: 原始输入样本
    target_labels: 目标序列标签
    model: 待攻击的模型
    confidence: 所需的置信度
    """
    output = model(data)
    


    data = data.clone().detach().requires_grad_(True)
    
    logits = DataUtils.greedy_decoder(output)
    logits = torch.tensor(logits).to(device)
    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

    log_probs = output.permute(2, 0, 1)
    log_probs = log_probs.log_softmax(2).requires_grad_()

    x = target_labels.numel()
    target_l = (x, )

    loss = ctc_loss(log_probs,
                        target_labels,
                        input_lengths=(18, ),
                        target_lengths=target_l)

    model.zero_grad()
    loss.backward(retain_graph=True)
    
    
    
    # 计算目标函数
    # 对于每个样本,计算当前预测标签和目标标签之间的距离
    try:
        dist = torch.sum(torch.abs(logits - target_labels))
    except RuntimeError:
        min_size = torch.minimum(torch.tensor(logits.size(-1)), torch.tensor(target_labels.size(-1)))
        logits = torch.narrow(logits, 0, 0, min_size)
        target_labels = torch.narrow(target_labels, 0, 0, min_size)
        dist = torch.sum(torch.abs(logits - target_labels))
        
    loss += torch.max(dist - confidence, torch.tensor(0.))

    # 加入 L2 正则化项
    l2_norm = torch.sum(torch.abs(data - data.detach()))
    return loss + 0.1 * l2_norm

def generate_adversarial_sample(x, target_labels, model, confidence=0.1, max_iter=100, lr=0.01):
    """
    生成对抗性样本
    x: 原始输入样本
    target_labels: 目标序列标签
    model: 待攻击的模型
    confidence: 所需的置信度
    max_iter: 最大迭代次数
    lr: 学习率
    """

    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    
    optimizer = optim.Adam([x_adv], lr=lr)
    
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = c_w_loss(x_adv, target_labels, model, confidence)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    return x_adv.detach()

#标签转换
def trans_label(label):
    lb = ""
    for i in label:
        lb += CHARS[i]
    return lb


batch_size = args.batch_size
img_dirs = os.path.expanduser(args.attack_img_dirs)
test_dataset = LPRDataLoader(img_dirs.split(','), args.img_size,
                             args.lpr_max_len)

def test(model, device, test_loader):

    epoch_size = len(test_dataset)
    batch_iterator = iter(
        torch.utils.data.DataLoader(test_dataset,
                                    batch_size,
                                    shuffle=True,
                                    collate_fn=DataUtils.collate_fn))

    # 循环遍历测试集中的所有示例
    for i in range(epoch_size):
        data, target, lengths = next(batch_iterator)

        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)

       
        # 对输入图像应用C&W攻击
        perturbed_data = generate_adversarial_sample(data, target, model, confidence=0.1, max_iter=100, lr=0.01)
        # 重新分类受扰乱的图像
        output = model(perturbed_data)

        # 保存生成图像


        final_pred = DataUtils.greedy_decoder(output)
        final_pred = torch.tensor(final_pred).to(device)
        image_array = perturbed_data.squeeze().detach().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)
        image_array = image_array / 0.0078125
        image_array += 127.5
        image_array = np.clip(image_array,0,255).astype(np.uint8)
        image = Image.fromarray(image_array)
        #image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image_array.transpose(1, 2, 0))
        image = image.convert('RGB')
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
        lb1 = target.tolist()
        lbx = [int(x) for x in lb1]
        lb2 = trans_label(lbx)
        file_name = lb2+'.png'
        folder_path = r'./C&W Adversarial examples'
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        image.save(file_path)
        #cv2.imencode('.jpg', image_array)[1].tofile(file_path)
    
test(lprnet, device, LPRDataLoader)