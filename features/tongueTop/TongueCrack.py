"""
作者：无
功能：舌体裂纹检测
"""
import os
import glob

__all__ = ['TongueCrack']

class TongueCrack():
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
                    "crack": 0, # 裂纹
                    "crack_index": 0 # 裂纹指数
                }
        """
        pass