import os
from PIL import Image
import cv2
import numpy as np
import torch
import importlib
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model.LPRNet import build_lprnet
from attack import get_parser
from data.load_data import CHARS, LPRDataLoader
import models
import shutil

module = importlib.import_module('.data_util', package='util')
DataUtils = getattr(module, 'DataUtils')
args = get_parser()
use_cuda=True
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = r"weights\Final_LPRNet_model.pth"
targeted_model = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
targeted_model = targeted_model.to(device)

def AdvGan_process_image(img_dir):

    image_nc=3
    gen_input_nc = image_nc

    # load the generator of adversarial examples
    pretrained_generator_path = r'weights\netG_epoch_60.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    image_path = os.path.join(os.getcwd(), img_dir)
    img = Image.open(image_path)
    img = transforms.ToTensor()(img).to(device)

    test_img = img.unsqueeze(0)
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
    image_array = image_array.transpose(1, 2, 0)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # 获取路径最后一个 / 的名字
    dir_name = os.path.basename(os.path.normpath(img_dir))
    lb2 = dir_name
    file_name = lb2
    folder_path = r'static/advGan_img'
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + '/' + file_name
    cv2.imencode('.jpg', image_array)[1].tofile(file_path)


    return file_path

def FGSM_process_image(img_dir, eps):
    tempdir="./static/temp_img"
    os.makedirs(tempdir, exist_ok=True)
    # 获取路径最后一个 / 的名字
    dir_name = os.path.basename(os.path.normpath(img_dir))
    dst_file = os.path.join('./static/temp_img', dir_name)
    shutil.copy(img_dir, dst_file)

    epsilon = float(eps)
    list_images = []
    list_images.append(tempdir)
    test_dataset = LPRDataLoader(list_images, args.img_size, args.lpr_max_len)
    
    
    img, target, length = test_dataset[0]
    shutil.rmtree(tempdir)
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.eval()


    image_path = os.path.join(os.getcwd(), img_dir)
    #img = Image.open(image_path)
    img = transforms.ToTensor()(img).to(device)
    target = torch.tensor(target).to(device)
    test_img = img.unsqueeze(0)
    test_img = test_img.permute(0, 2, 3, 1)
    # 设置张量的requires_grad属性
    test_img.requires_grad = True

    output = lprnet(test_img)
    init_pred = DataUtils.greedy_decoder(output)
    init_pred = torch.tensor(init_pred).to(device)
    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')
    log_probs = output.permute(2, 0, 1)
    log_probs = log_probs.log_softmax(2).requires_grad_()

    x = target.numel()
    target_l = (x, )

    loss = ctc_loss(log_probs,
                    target,
                    input_lengths=(18, ),
                    target_lengths=target_l)

    lprnet.zero_grad()
    loss.backward()

    data_grad = test_img.grad.data
    perturbed_data = fgsm_attack(test_img, epsilon, data_grad)
    # 重新分类受扰乱的图像
    output = lprnet(perturbed_data)

    #save
    image_array = perturbed_data.squeeze().detach().cpu().numpy()
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array.transpose(1, 2, 0))
    image = image.convert('RGB')
    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))
    # 获取路径最后一个 / 的名字
    #dir_name = os.path.basename(os.path.normpath(img_dir))
    lb2 = dir_name
    file_name = lb2
    folder_path = r'static/adv_fsgm_img'
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + '/' + file_name
    file_path = add_marker_to_extension(file_path, '-' + str(eps))
    image.save(file_path)

    return file_path

# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + (epsilon * sign_data_grad)
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


def PGD_process_image(img_dir, eps, iters):
    tempdir="./static/temp_img"
    os.makedirs(tempdir, exist_ok=True)
    # 获取路径最后一个 / 的名字
    dir_name = os.path.basename(os.path.normpath(img_dir))
    dst_file = os.path.join('./static/temp_img', dir_name)
    shutil.copy(img_dir, dst_file)

    epsilon = float(eps)
    iters = int(iters)
    list_images = []
    list_images.append(tempdir)
    test_dataset = LPRDataLoader(list_images, args.img_size, args.lpr_max_len)
    
    
    img, target, length = test_dataset[0]
    shutil.rmtree(tempdir)
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.eval()


    image_path = os.path.join(os.getcwd(), img_dir)
    #img = Image.open(image_path)
    img = transforms.ToTensor()(img).to(device)
    target = torch.tensor(target).to(device)
    test_img = img.unsqueeze(0)
    test_img = test_img.permute(0, 2, 3, 1)
    # 设置张量的requires_grad属性
    test_img.requires_grad = True

    output = lprnet(test_img)
    init_pred = DataUtils.greedy_decoder(output)
    init_pred = torch.tensor(init_pred).to(device)
    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')
    log_probs = output.permute(2, 0, 1)
    log_probs = log_probs.log_softmax(2).requires_grad_()

    x = target.numel()
    target_l = (x, )

    loss = ctc_loss(log_probs,
                    target,
                    input_lengths=(18, ),
                    target_lengths=target_l)

    lprnet.zero_grad()
    loss.backward()

    data_grad = test_img.grad.data
    # 对输入图像应用PGDs攻击
    perturbed_data = pgd_attack(lprnet, test_img, epsilon, iters, data_grad, target)
    # 重新分类受扰乱的图像
    output = lprnet(perturbed_data)

    #save
    image_array = perturbed_data.squeeze().detach().cpu().numpy()
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array.transpose(1, 2, 0))
    image = image.convert('RGB')
    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))
    # 获取路径最后一个 / 的名字
    #dir_name = os.path.basename(os.path.normpath(img_dir))
    lb2 = dir_name
    file_name = lb2
    folder_path = r'static/adv_pgd_img'
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + '/' + file_name
    file_path = add_marker_to_extension(file_path, '-' + str(eps) + '-' + str(iters))
    image.save(file_path)

    return file_path

def pgd_attack(model, image, epsilon, iters, data_grad, target):
    alpha = epsilon / iters
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True

    for i in range(iters):
        #perturbed_image.requires_grad = True
        # 通过模型前向传递扰动后的图像
        output = model(perturbed_image)

        # 计算损失
        ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

        log_probs = output.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        x = target.numel()
        target_l = (x, )

        loss = ctc_loss(log_probs, target, input_lengths=(18, ), target_lengths=target_l)

        # 对图像进行反向传播，计算梯度
        model.zero_grad()
        loss.backward()

        # 从数据梯度中获取符号
        sign_data_grad = data_grad.data.sign()

        # 对图像进行扰动
        perturbed_image = perturbed_image + alpha * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, min=image - epsilon, max=image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach().requires_grad_(True)


    return perturbed_image

def CW_process_image(img_dir, iters, confidence, c):
    iters=int(iters)
    confidence=float(confidence)
    c=float(c)
    tempdir="./static/temp_img"
    os.makedirs(tempdir, exist_ok=True)
    # 获取路径最后一个 / 的名字
    dir_name = os.path.basename(os.path.normpath(img_dir))
    dst_file = os.path.join('./static/temp_img', dir_name)
    shutil.copy(img_dir, dst_file)

    list_images = []
    list_images.append(tempdir)
    test_dataset = LPRDataLoader(list_images, args.img_size, args.lpr_max_len)
    img, target, length = test_dataset[0]
    shutil.rmtree(tempdir)
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.eval()

    img = torch.tensor(img).to(device)
    target = torch.tensor(target).to(device)
    img = img.unsqueeze(0)

    # 对输入图像应用C&W攻击
    perturbed_data = generate_adversarial_sample(img, target, lprnet, confidence=confidence, max_iter=iters, c=c, lr=0.01)
    # 重新分类受扰乱的图像
    output = lprnet(perturbed_data)

    # 保存生成图像


    final_pred = DataUtils.greedy_decoder(output)
    final_pred = torch.tensor(final_pred).to(device)
    image_array = perturbed_data.squeeze().detach().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)
    image_array = image_array / 0.0078125
    image_array += 127.5
    image_array = np.clip(image_array,0,255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image = image.convert('RGB')
    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))
    lb2 = dir_name
    file_name = lb2
    folder_path = r'static/adv_cw_img'
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + '/' + file_name + '-' + str(iters) + '-' + str(c)
    file_path = add_marker_to_extension(file_path, '-' + str(iters)+'-'+ str(confidence)+'-'+ str(c))
    print(file_path)
    image.save(file_path)
    return file_path

def generate_adversarial_sample(x, target_labels, model, confidence, max_iter, c , lr=0.01, ):
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
        loss = c_w_loss(x_adv, target_labels, model, confidence, c)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    return x_adv.detach()

def c_w_loss(data, target_labels, model, confidence, c):
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
    return loss + c * l2_norm

def add_marker_to_extension(file_path, marker):
    # 获取文件名和扩展名
    filename, ext = os.path.splitext(file_path)
    
    # 拼接新的扩展名
    new_ext = f"{marker}{ext}"
    
    # 拼接新的文件路径
    new_file_path = f"{filename}{new_ext}"
    
    return new_file_path