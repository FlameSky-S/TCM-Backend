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

from .TongueBottomSeg import tongueBottomSeg
from .VeinsSeg import veinSeg
from .VeinsColor import veinColor
from .VeinsIndex import veinIndex
from utils.tools import img_to_base64

__all__ = ['tongueBottomFeature']

class tongueBottomFeature():
    def __init__(self):
        self.tongue_bottom_seg = tongueBottomSeg(is_train=False, gpu_device_id=0, use_pretrained=True, pretrained_path='Default')
        self.vein_seg = veinSeg(is_train=False, gpu_device_id=0, use_pretrained=True, pretrained_path='Default')
        self.vein_color = veinColor()
        self.vein_index = veinIndex()
        # 功能映射
        self.METHODS_MAP = {
            "TONGUE_BOTTOM_SEG": self.tongue_bottom_seg.get,
            "VEIN_SEG": self.vein_seg.get,
            "VEIN_COLOR": self.vein_color.get,
            "VEIN_INDEX": self.vein_index.get
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
        tongue_ret = self.tongue_bottom_seg.get(raw_img)
        tongue_img = tongue_ret['imageData']
        ret = {}
        for m in METHODS:
            if m == 'TONGUE_BOTTOM_SEG':
                ret[m] = tongue_ret
            else:
                # 使用分割后的舌下部图片
                ret[m] = self.METHODS_MAP[m](tongue_img)
        return ret

if __name__ == "__main__":
    pass