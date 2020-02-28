from easydict import EasyDict as edict 
import torch


config = edict()
config.device = torch.device('cuda')

# model setting
config.model_head = {'hm':2,'reg':2,'wh':2}
config.model_layer = 10
config.down_ratio = 4
config.head_conv = 64

# basic setting
config.mean = [0.485, 0.456, 0.406]
config.std = [0.229, 0.224, 0.225]
config.num_classes = 2
config.input_h = 512
config.input_w = 512
config.topk = 100
config.vis_threshold = 0.4
config.dict = {'0':'face_mask', '1':'face'}

