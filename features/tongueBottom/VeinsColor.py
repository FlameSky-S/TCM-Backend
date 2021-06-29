"""
作者：李华东
功能：舌下脉络颜色分类
"""

__all__ = ['VeinsColor']

class VeinsColor():
    def __init__(self):
        pass

    def do(self, src_path, dst_path=None):
        """
        function:
            单张图片舌下脉络颜色分类
        parameters:
            src_path: 图片路径
            dst_path: 结果保存路径，若为空，则不保存（保存成相应的json文件）
        return:
            res: 结果数据，返回字典
                {
                    "loc1":{
                        "labelname": "左侧",
                        "colorValue": [255, 255, 255] (L, A, B)
                    },
                    "loc2":{
                        "labelname": "右侧",
                        "colorValue": [155, 155, 155]
                    },
                    "imageData": "base64编码, 见utils.tools.img_to_txt",
                    "imageSize": [100, 100]
                }
        """
        pass