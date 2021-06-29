import os
import cv2
from glob import glob

class TongueBottomPre():
    def __init__(self):
        pass

    def __remove_lights(self, raw_img, dst_img):
        """
        function:
            去除图像中的反光躁点
        parameters:
            raw_img: 原图片
            dst_img: 目标图片
        """
        _, mask = cv2.threshold(raw_img, 200, 255, cv2.THRESH_BINARY)
        dst_img = cv2.inpaint(raw_img, mask, 10, cv2.INPAINT_TELEA)
        return dst_img

    def doSingle(self, src_path, dst_path=None, save_size=(100,100)):
        """
        function:
            人脸检测，从含背景的图片中提取人脸部分
        parameters:
            src_path: 原图片路径
            dst_path: 结果保存路径，若为空，则不保存图片
            save_size: 图片保存的大小
        return:
            img: 图片数据
        """
        pass

    def doBatch(self, src_dir, dst_dir, save_size=(100, 100)):
        """
        function:
            批量处理图片，将处理结果保存在dst_dir目录中
        parameters:
            src_dir: 原图片目录，内部可分目录存放（用glob遍历图片绝对路径）
            dst_dir: 存放目录
            save_size: 图片保存的大小
        return:
            无
        """
        pass