"""
作者：常然
功能：舌苔颜色分类 & LAB值
"""
import os
import glob

__all__ = ['CoatingColor']

class CoatingColor():
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
                    "coating_color": 0, # 苔色
                    "coating_L_value": 0, # L
                    "coating_A_value": 0, # A
                    "coating_B_value": 0, # B
                }
        """
        pass