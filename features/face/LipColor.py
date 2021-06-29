"""
作者：无
功能：嘴唇颜色分类 & LAB值
"""
import os
import glob

__all__ = ['LipColor']

class LipColor():
    def __init__(self):
        pass

    def do(self, src_path):
        """
        function:
            处理单张人脸图片
        parameters:
            src_path: 人脸图片路径（无其他背景的纯人脸图片）
        return:
            res: 结果数据，返回字典
                {
                    "lip_color": 0, # 嘴唇颜色
                    "lip_L_value": 0, # L值
                    "lip_A_value": 0, # A值
                    "lip_B_value": 0, # B值
                }
        """
        pass