"""
作者：无
功能：舌体裂纹检测
"""
import os

__all__ = ['tongueCrack']

class tongueCrack():
    def __init__(self):
        pass

    def get(self, img):
        """
        function:
            舌体裂纹检测（在舌体分割之后执行）
        parameters:
            img: 舌体图片
        return:
            res: 结果数据，返回字典
                {
                    "crack": 0, # 裂纹
                    "crack_index": 0 # 裂纹指数
                }
        """
        pass