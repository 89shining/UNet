import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#-------------------------------------------------------------#
#   UNet 预测：支持灰度掩码输出（背景=0，目标=255）
#-------------------------------------------------------------#
class Unet(object):
    _defaults = {
        "model_path"    : 'logs/best_epoch_weights.pth',
        "num_classes"   : 2,
        "backbone"      : "resnet50",
        "input_shape"   : [512, 512],
        "cuda"          : True,
        "gray_output"   : False,   # ✅ 灰度输出模式
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()
        show_config(**self._defaults)

    #-----------------------------------------#
    #   构建模型并加载权重
    #-----------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f"{self.model_path} model loaded.")
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #-----------------------------------------#
    #   核心预测函数：输出灰度掩码
    #-----------------------------------------#
    def detect_image(self, image):
        """
        输入：PIL图像
        输出：灰度掩码（前景255，背景0）
        """
        image = cvtColor(image)
        orininal_h, orininal_w = np.array(image).shape[:2]

        # resize + 归一化
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 去除padding区域
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]

            # resize回原尺寸
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # 取类别标签
            pr = pr.argmax(axis=-1)

        # ✅ 仅输出灰度掩码（背景=0，前景=255）
        mask = (pr > 0).astype(np.uint8) * 255
        return Image.fromarray(mask)
