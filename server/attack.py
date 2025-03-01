import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--attack_img_dirs', default=r"data\validation", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=7, help='license plate number max length.')
    parser.add_argument('--batch_size', default=1, help='batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--attack_model', default='./weights/Final_LPRNet_model.pth', help='被攻击模型')
    
    args = parser.parse_args()

    return args
    
