"""
作者：常然
功能：舌体（舌质）颜色分类 & LAB值
"""
import os
import glob

__all__ = ['TongueColor']

class TongueColor():
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
                    "tongue_color": 0, # 舌色
                    "tongue_L_value": 0, # L
                    "tongue_A_value": 0, # A
                    "tongue_B_value": 0, # B
                }
        """
        pass