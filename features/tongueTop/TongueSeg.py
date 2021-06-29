"""
作者：常然
功能：舌体提取
"""
import os
import glob

__all__ = ['TongueSeg']

class TongueSeg():
    def __init__(self):
        pass

    def do(self, src_path, dst_path=None):
        """
        function:
            处理单张图片
        parameters:
            src_path: 图片路径（无其他背景的纯人脸图片）
            dst_path: 结果保存路径，若为空，则不保存（保存成相应的json文件）
        return:
            res: 结果数据，返回字典
                {
                    "colorValue": [255, 255, 255]
                    "imageData": "base64编码, 见utils.tools.img_to_txt",
                    "imageSize": [100, 100]
                }
        """
        pass