from ..utils.config import config
from ..utils.util import load_model, ctdet_post_process, draw_bbox
from .mobilenet_v2 import get_mobile_net
from ..utils.image import get_affine_transform
from .decode import ctdet_decode
import numpy as np 
import torch
import time
import cv2


class FaceMaskDetector(object):
    '''
    init the model

    args:
        config: the default config setting
        device: the inference device

    return:
        None
    '''
    def __init__(self, model_path, conf=config, device=None):
        self.device = device if device is not None else conf.device
        
        self.model = get_mobile_net(conf.model_layer, conf.model_head, conf.head_conv)
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = np.array(conf.mean, dtype=np.float32).reshape(1,1,3)
        self.std = np.array(conf.std, dtype=np.float32).reshape(1, 1, 3)
        self.num_classes = conf.num_classes
        self.input_h = conf.input_h
        self.input_w = conf.input_w
        self.down_ratio = conf.down_ratio
        self.topk = conf.topk
        self.vis_threshold = conf.vis_threshold
        self.dict = conf.dict

    def pre_process(self, image, meta=None):
        '''
        for pre_process the images,fix the input size

        args:
            image:cv2
            meta:none
        
        return:
            image: torch image
            meta: the scale information
        '''
        height, width = image.shape[0:2]
        new_height = int(height)
        new_width  = int(width)
        inp_height, inp_width = self.input_h, self.input_w      
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height),flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 
                'out_height': inp_height // self.down_ratio, 
                'out_width': inp_width // self.down_ratio}
        return images, meta

    def __call__(self, image_or_path, meta=None, return_time=False):
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif type(image_or_path) == type (''): 
            image = cv2.imread(image_or_path)

        images, meta = self.pre_process(image, meta)
        images = images.to(self.device)
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] 
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, K=self.topk) 

        dets = self.post_process(dets, meta)
        dets = self.threshold_process(dets)

        if return_time:
            return dets, forward_time
        else:
            return dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def threshold_process(self, dets):
        result = []
        for j in range(1, self.num_classes + 1):
            for bbox in dets[j]:
                if bbox[4] > self.vis_threshold:
                    result.append([bbox[:4], bbox[4], self.dict[str(j-1)] ])
        return result
        

    def detect_image_show(self, image):
        image = cv2.imread(image)
        bbox = self.__call__(image)
        draw_bbox(image, bbox)
        return bbox

    def detect_videoshow(self, video_path):
        print('TO DO')
        pass
    