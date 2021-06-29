"""
作者：常然
已实现功能：舌体提取 & 齿痕检测 & (舌苔 & 舌质 -> 分离 & 颜色分类）
"""
import os
import cv2
import pickle
import collections
import numpy as np
from sklearn.cluster import KMeans

from tqdm import tqdm
from glob import glob
from PIL import Image
from io import BytesIO

from .tongueSeg import tongueSeg
from .toothMark import toothMark
from .tongueColor import tongueColor
from .tongueCrack import tongueCrack
from .tongueWidth import tongueWidth
from .tongueThickness import tongueThickness
from utils.tools import img_to_base64

__all__ = ['tongueTopFeature']

class tongueTopFeature():
    def __init__(self, min_pixel_per_tm = 50, alpha = 0.04, beta = 4):
        '''
        min_pixel_per_tm: 齿痕区域占多少个像素时被判断为是一块齿痕
        alpha: 齿痕区域占比超过百分之alpha时则认为有齿痕
        beta: 齿痕块数超过beta时则认为有齿痕
        '''
        self.tongue_seg = tongueSeg(is_train=False, gpu_device_id=0, use_pretrained=True, pretrained_path='Default')
        self.tooth_mark = toothMark(is_train=False, gpu_device_id=0, use_pretrained=True, pretrained_path='Default', min_pixel_per_tm=min_pixel_per_tm, alpha=alpha, beta=beta)
        self.tongue_color = tongueColor()
        self.tongue_crack = tongueCrack()
        self.tongue_width = tongueWidth()
        self.tongue_thickness = tongueThickness()
        # 功能映射
        self.METHODS_MAP = {
            "TONGUE_SEG": self.tongue_seg.get,
            "TOOTH_MARK": self.tooth_mark.get,
            "TONGUE_COLOR": self.tongue_color.get,
            "Crack": self.tongue_crack.get,
            "Width": self.tongue_width.get,
            "Thickness": self.tongue_thickness.get
        }

    def __getImageFromMask(self, raw_img, msk):
        '''
        将灰度图像结果叠加到原始图像上，相当于一层蒙板盖在原始图像上，也可增加透明效果
        '''
        msk = msk > 200
        res = np.zeros(raw_img.shape)
        for i in range(3):
            res[:,:,i] = msk * raw_img[:,:,i]
        return res
    
    def doAll(self, src_path, METHODS=[]):
        """
        function:
            处理单张图片
        parameters:
            src_path: 图片路径（无其他背景的纯人脸图片）
        return:
            res: 结果数据，返回字典
                {
                    "TONGUE_SEG": {
                        "colorValue": [255, 255, 255]
                        "imageData": "base64编码, 见utils.tools.img_to_txt",
                        "imageSize": [100, 100]
                    },
                    ...
                }
        """
        if METHODS == []:
            return {}
        raw_img = cv2.imread(src_path)
        tongue_ret = self.tongue_seg.get(raw_img)
        tongue_img = tongue_ret['imageData']
        ret = {}
        for m in METHODS:
            if m == 'TONGUE_SEG':
                ret[m] = tongue_ret
            elif m == "TOOTH_MARK":
                ret[m] = self.METHODS_MAP[m](raw_img, tongue_img)
            else:
                # 使用分割后的舌体图片
                ret[m] = self.METHODS_MAP[m](tongue_img)
        return ret

if __name__ == "__main__":
    pass