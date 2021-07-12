from re import A
from sklearn.utils import tosequence
from features.pulse import hemodynamics, pulseFeature
from data.dataset.features import faceDataset, tongueTopDataset
from features.face.AreaSeg import AreaSeg
from features.tongueTop import tongueColor, tongueSeg
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import re


# hemo = hemodynamics()
# res = hemo.get('test_data/pulse_2.txt')
# print(res)
# p = pulseFeature()
# p.doSingle('/home/sharing/disk3/Datasets/TCM-Datasets/陕西藏族汉族脉诊数据/脉诊/raw/20180911001.txt', 'tmp', METHODS=['hemodynamics'])
# p.doDataset('/home/sharing/disk3/Datasets/TCM-Datasets/陕西藏族汉族脉诊数据/脉诊/raw', 'tmp', ['hemodynamics'])


# model = AreaSeg(gpu_device_id=1)
# img = Image.open('/home/sharing/disk3/Datasets/TCM-Datasets/多模态数据/面诊/raw/27777.jpg')
# # img = img.resize((512,512))
# # img = Image.open('参考/面部语义分割/data/imgs/0.png')
# output = model.do_test(img)
# print(output)


# model = tongueSeg()
# img = Image.open("test_data/tongueTop_21007.jpg")
# model.get(img)


def getImageFromMask(raw_img, msk):
    msk = msk > 200
    res = np.zeros(raw_img.shape)
    for i in range(3):
        res[:,:,i] = msk * raw_img[:,:,i]
    res = res.astype('uint8')
    return res

# dataset = tongueTopDataset('/home/sharing/disk3/Datasets/TCM-Datasets/DemoTraining')
# train_set, dev_set, _ = torch.utils.data.random_split(dataset, [800, 200, 7025])
# model_seg = tongueSeg()
# model_cluster = tongueColor()

# tongue_list = []
# label_list = []
# for data in tqdm(train_set):
#     out = model_seg.get(data['image'])
#     res = getImageFromMask(np.asarray(data['image']), out)
#     tongue, coat = model_cluster.cluster(res)
#     tongue_vec = model_cluster.getVecFromImage(np.asarray(data['image']), tongue)
#     # coat = getVecFromImage(np.asarray(data['image']), coat)
#     tongue_list.append(tongue_vec)
#     label_list.append(data['tongue_color'])

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=100)
# clf = clf.fit(tongue_list, label_list)

# tongue_test_list = []
# label_test_list = []
# for data in tqdm(dev_set):
#     out = model_seg.get(data['image'])
#     res = getImageFromMask(np.asarray(data['image']), out)
#     tongue, coat = model_cluster.cluster(res)
#     tongue_vec = model_cluster.getVecFromImage(np.asarray(data['image']), tongue)
#     # coat = getVecFromImage(np.asarray(data['image']), coat)
#     tongue_test_list.append(tongue_vec)
#     label_test_list.append(data['tongue_color'])

# y_pred = clf.predict(tongue_test_list)
# evaluate = y_pred == label_test_list
# print(evaluate.sum())


model_seg = tongueSeg()
model_cluster = tongueColor()
for img_path in tqdm(glob('/home/sharing/disk3/Datasets/TCM-Datasets/DemoTraining/舌上/processed/*.jpg')):
    filename = re.search('([^<>/\\\|:""\*\?]+)\.\w+$', img_path).group()
    img = Image.open(img_path)
    out = model_seg.get(img)
    res = getImageFromMask(np.asarray(img), out)
    Image.fromarray(res).save(f'/home/sharing/disk3/Datasets/TCM-Datasets/DemoTraining/舌上/temp/seg1/{filename}.jpg')