
import argparse
import cv2
import os
import time
import numpy as np
import torch
from PIL import Image
import imageio
from dataset.transforms import BaseTransform
# from utils.misc import load_weight
from config import build_dataset_config, build_model_config
from models.detector import build_model
import imageio
import psutil


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.35, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')

    # model
    parser.add_argument('-v', '--version', default='yowo', type=str,
                        help='build YOWO')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    return parser.parse_args()


def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    
    #initialiser les pred.wegth et pred.bias de la couche de sortie 
    checkpoint_state_dict['pred.bias']=model_state_dict['pred.bias']
    checkpoint_state_dict['pred.weight']=model_state_dict['pred.weight']
    
    # check
    i=0
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
                # checkpoint_state_dict[k]=shape_model[k]
                # i=i+1
                # print("************************",i)
        else:
            checkpoint_state_dict.pop(k)
            # print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model                    
if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = 2

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    
    model_path = os.path.join("/kaggle/working/", 'modelform0.pth')
    torch.save(model, model_path)

    
    # # load trained weight
    modelwithweight = load_weight(model=model, path_to_ckpt=args.weight)
    save_model_path = os.path.join("/kaggle/working/", 'model99999.pth')

    # Enregistrez le modèle
    torch.save(modelwithweight, save_model_path)
    # to eval
    model = model.to(device).eval()
 
