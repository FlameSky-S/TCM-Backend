import os
from glob import glob

class PulsePre():
    def __init__(self):
        pass

    def doSingle(self, src_path, dst_path=None, save_size=(200, 50)):
        """
        function:
            将脉诊数值转换成波形图片
        parameters:
            src_path: 脉诊文件路径（*.txt）
            dst_path: 结果保存路径，若为空，则不保存
            save_size: 图片保存的大小
        return:
            img: 波形图片数据
        """
        pass

    def doBatch(self, src_dir, dst_dir, save_size=(100, 100)):
        """
        function:
            批量处理脉诊数据，将处理结果保存在dst_dir目录中
        parameters:
            src_dir: 原脉诊数据目录，内部可分目录存放（用glob遍历文件绝对路径）
            dst_dir: 存放目录
            save_size: 图片保存的大小
        return:
            无
        """
        pass