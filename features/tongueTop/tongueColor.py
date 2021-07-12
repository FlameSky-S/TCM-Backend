"""
作者：常然
功能：舌苔质分离 & 颜色分类
方法：传统机器学习
"""
import os
import glob
import cv2
import base64
import pickle
import collections
import numpy as np
from sklearn.cluster import KMeans

from tqdm import tqdm
from PIL import Image

class tongueColor():
    """
    舌苔质分离 & 获取舌苔舌质颜色
    """
    def __init__(self):
        self.classifier = KMeans(n_clusters=3)
    
    # def do_train(self, train_x, train_y, classifier):
    #     X = np.load(train_x)
    #     Y = np.load(train_y)   
    #     clf = classifier.fit(X, Y)
    #     return clf

    def cluster(self, img):
        """
        分离舌苔舌质
        img为np array
        """
        h,w,_ = img.shape
        vec = np.reshape(img,(h*w,3))
        label = self.classifier.fit_predict(vec)
        label = np.reshape(label, (h,w))
        start, end = 0, 0
        for i in range(h):
            classes = []
            for j in range(w):
                if label[i][j] not in classes:
                    classes.append(label[i][j])
            if len(classes) == 3:
                if start == 0:
                    start = i
            if len(classes) < 3 and start != 0 :
                end = i
                break
        ref = int((start+end)/2)
        mid = label[ref,:]
        fields = []
        for c in mid:
            if c not in fields:
                fields.append(c)
        if len(fields) < 3:
            fields = [-1,-1,-1] # bad data
        # 舌质
        mask_shezhi = (label == fields[1])
        shezhi = np.zeros(img.shape)
        for i in range(3):
            shezhi[:,:,i] = mask_shezhi * img[:,:,i]
        msk_shezhi = mask_shezhi*np.ones(mask_shezhi.shape)*255
        # 舌苔
        mask_shetai = (label == fields[2])
        shetai = np.zeros(img.shape)
        for i in range(3):
            shetai[:,:,i] = mask_shetai * img[:,:,i]
        msk_shetai = mask_shetai*np.ones(mask_shetai.shape)*255
        return msk_shetai, msk_shezhi
    
    def getVecFromImage(self, img, msk, outputlen=100):
        """
        获取图片msk区域的向量表示，等间隔采样100个点作为代表点
        """
        msk = msk > 200
        allpts = []
        for i in range(len(img)):
            for j in range(len(img[0])):
                if msk[i][j]:
                    allpts.append(img[i][j])
        length = len(allpts)
        if length == 0:
            return np.zeros(3*outputlen)
        interval = int(length / outputlen)
        interval = interval if interval != 0 else 1
        vec = []
        for i in range(length):
            if i % interval == 0:
                vec.append(allpts[i])
        vec = np.array(vec)
        tmp = vec
        while len(vec) < outputlen:
            vec = np.append(vec, tmp, axis=0)
        vec = vec[:outputlen]
        vec = np.append(vec[:,0], np.append(vec[:,1],vec[:,2]))
        # TODO: 考虑返回颜色均值
        return vec
    
    def __getLAB(self, img, msk):
        """
        获取区域的L，A，B值
        """
        pass
    
    def get(self, img):
        """
        parameters:
            img: 分割后的舌体图片
        return:
            res: 结果数据，返回字典
                {
                    "tongue_color": 0, # 舌苔颜色
                    "tongue_L_value": 0, # L
                    "tongue_A_value": 0, # A
                    "tongue_B_value": 0, # B
                    "coating_color": 0, # 苔色
                    "coating_L_value": -1, # L 暂无
                    "coating_A_value": -1, # A 暂无
                    "coating_B_value": -1, # B 暂无
                }
        """
        shetai_msk, shezhi_msk, _ = self.__cluster(img)
        # 舌苔颜色分类
        shetai_vec = self.__getVecFromImage(img, shetai_msk)
        with open(self.shetai_color_clf_path, 'rb') as f:
            clf = pickle.load(f)
            ans_shetai = clf.predict([shetai_vec])
        # 舌质颜色分类
        shezhi_vec = self.__getVecFromImage(img, shezhi_msk)
        with open(self.shezhi_color_clf_path, 'rb') as f:
            clf = pickle.load(f)
            ans_shezhi = clf.predict([shezhi_vec])
        res = {
            "tongue_color": ans_shezhi[0], # 舌质颜色
            "tongue_L_value": 0, # L
            "tongue_A_value": 0, # A
            "tongue_B_value": 0, # B
            "coating_color": ans_shetai[0], # 舌苔颜色
            "coating_L_value": 0, # L
            "coating_A_value": 0, # A
            "coating_B_value": 0, # B
        }
        return res

if __name__ == "__main__":
    pass