"""
作者：无
功能：舌体胖瘦检测
"""
import os

__all__ = ['tongueWidth']

class tongueWidth():
    def __init__(self):
        pass

    def get(self, img):
        """
        function:
            舌体胖瘦检测（在舌体分割之后执行）
        parameters:
            img: 舌体图片
        return:
            res: 结果数据，返回字典
                {
                    "tongue_width": 0 # 胖瘦指数
                }
        """
        pass