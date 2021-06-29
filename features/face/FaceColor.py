"""
作者：无
功能：人脸颜色分类 & LAB值特征提取
"""
import os
import glob

__all__ = ['FaceColor']

class FaceColor():
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
                    "face_color": 0, # 面部颜色
                    "face_L_value": 0, # L值
                    "face_A_value": 0, # A值
                    "face_B_value": 0, # B值
                }
        """
        pass