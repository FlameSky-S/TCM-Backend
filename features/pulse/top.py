"""
脉诊特征提取
已实现功能：血液动力学参数
待实现功能：脉象分类
"""

import numpy as np
from .hemodynamics import hemodynamics
from .pulseCondition import pulseCondition
from glob import glob
import os
import re
import json


class pulseFeature():
    def __init__(self):
        self.hemodynamics = hemodynamics()
        # self.pulse_condition = pulseCondition()
        self.METHODS_MAP = {
            "hemodynamics": self.hemodynamics.get,
            # "pulse_condition": self.pulse_condition.get
        }

    def doSingle(self, src_path, tmp_path, METHODS=[]):
        """
        function:
            处理单条数据
        parameters:
            src_path: 数据文件路径（txt）
            tmp_path: 结果及中间结果存放路径
            METHODS: 需要调用的方法列表，取值为self.METHODS_MAP中的键值
        return:
            res: 结果数据，返回字典
                {
                    "HEMO": {
                        "avg_period": np.array([0,0,...,0])
                        "period_feature": [0.561, 0.333, 0.003, 0.775, 0.171, 0.164],
                        "hemo_feaure": [6.0, 0.0]
                    },
                    ...
                }
        """
        if METHODS == []:
            return {}
        
        res = {}
        for m in METHODS:
            res[m] = self.METHODS_MAP[m](src_path, tmp_path)

        return res

    def doDataset(self, src_path, tmp_path, METHODS=[]):
        """
        function:
            处理数据集，不保存中间结果，返回结果文件下载地址
        parameters:
            src_path: 数据集路径
            tmp_path: 结果数据存放路径
            METHODS: 需要调用的方法列表，取值为self.METHODS_MAP中的键值
        return:
            res: 结果数据下载地址
        """

        if METHODS == []:
            return ''

        final_result = []
        for file in glob(os.path.join(src_path, "*.txt")):
            res = {}
            res['id'] = re.search('([^<>/\\\|:""\*\?]+)\.\w+$', file).group()
            for m in METHODS:
                res[m] = self.METHODS_MAP[m](file)
            final_result.append(res)
        
        save_path = os.path.join(tmp_path, 'pulse.json')
        with open(save_path, 'w') as f:
            json.dump(final_result, f)

        return save_path