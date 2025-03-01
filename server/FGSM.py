from __future__ import print_function
import os
import cv2
import torch
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
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
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

# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


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


def test(model, device, test_loader, epsilon):
    # 精度计数器
    correct = 0
    adv_examples = []

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

        # 设置张量的requires_grad属性
        data.requires_grad = True

        # 通过模型前向传递数据
        output = model(data)
        init_pred = DataUtils.greedy_decoder(output)
        init_pred = torch.tensor(init_pred).to(device)
        try:
            if not torch.all(torch.eq(init_pred, target)):
                continue
        except RuntimeError:
            pass

        ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

        log_probs = output.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        x = target.numel()
        target_l = (x, )

        loss = ctc_loss(log_probs,
                        target,
                        input_lengths=(18, ),
                        target_lengths=target_l)

        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        # 对输入图像应用FGSM攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # 重新分类受扰乱的图像
        output = model(perturbed_data)

        # 保存生成图像

        final_pred = DataUtils.greedy_decoder(output)
        final_pred = torch.tensor(final_pred).to(device)
        image_array = perturbed_data.squeeze().detach().cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array.transpose(1, 2, 0))
        image = image.convert('RGB')
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
        lb1 = target.tolist()
        lbx = [int(x) for x in lb1]
        lb2 = trans_label(lbx)
        file_name = lb2 + '.jpg'
        x = str(epsilon)
        folder_path = r'./PGD Adversarial examples' + '/' + x
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        image.save(file_path)

"""         try:
            if torch.eq(final_pred, target).all():
                correct += 1
                # 保存0 epsilon示例的特例
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred, final_pred, adv_ex))
            else:
                # 稍后保存一些用于可视化的示例
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred, final_pred, adv_ex))
        except RuntimeError:
            pass

    # 计算这个epsilon的最终准确度
    final_acc = correct / float(epoch_size)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
        epsilon, correct, epoch_size, final_acc))

    # 返回准确性和对抗性示例
    return final_acc, adv_examples

accuracies = []
examples = [] """

# 对每个epsilon运行测试
for eps in epsilons:
    #acc, ex = test(lprnet, device, LPRDataLoader, eps)
    # accuracies.append(acc)
    # examples.append(ex)
    test(lprnet, device, LPRDataLoader, eps)

""" plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# 在每个epsilon上绘制几个对抗样本的例子
cnt = 0
plt.figure(figsize=(8, 10))




font = FontProperties(fname="language/simhei.ttf")
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=12)
        orig, adv, ex = examples[i][j]
        orig_lb = trans_label(orig)
        adv_lb = trans_label(adv)
        plt.title("{} -> {}".format(orig_lb, adv_lb), fontproperties=font, fontsize=9)
        ex = ex.transpose(1, 2, 0)
        image_rgb = cv2.cvtColor(ex, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)

plt.tight_layout()
plt.show() """
