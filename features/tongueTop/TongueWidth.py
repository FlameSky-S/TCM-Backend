"""
作者：无
功能：舌体胖瘦检测
"""
import os
import glob

__all__ = ['TongueWidth']

class TongueWidth():
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
                    "tongue_width": 0 # 胖瘦指数
                }
        """
        pass