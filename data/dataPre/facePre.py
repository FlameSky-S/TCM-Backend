import os
from glob import glob

class FacePre():
    def __init__(self):
        pass

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