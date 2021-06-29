"""
作者：常然
功能：舌体齿痕检测
"""
import os
import glob

__all__ = ['ToothMark']

class ToothMark():
    def __init__(self):
        pass

    def do(self, src_path):
        """
        function:
            处理单张舌体图片
        parameters:
            src_path: 舌上图片路径
        return:
            res: 结果数据，返回字典
                {
                    "tooth_mark": 0 # 齿痕检测
                }
        """
        pass