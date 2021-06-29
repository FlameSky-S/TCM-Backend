"""
作者：无
功能：脸部光泽指数 & 有光泽指数 & 无光泽指数 & 少光泽指数
"""
import os
import glob

__all__ = ['FaceLight']

class FaceLight():
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
                    "face_light_result": 0, # 光泽判断结果
                    "face_light_index": 0, # 有光泽指数
                    "face_less_light_index": 0, # 少光泽指数
                    "face_no_light_index": 0 # 无光泽指数
                }
        """
        pass