import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2

def cluster(img, classifier=KMeans(n_clusters=3)):
    # img = cv2.imread(image)/255
    h,w,_ = img.shape
    vec = np.reshape(img,(h*w,3))
    label = classifier.fit_predict(vec)
    label = np.reshape(label, (h,w))
    cv2.imwrite('test.jpg', label*255/5)
    start = 0
    end = 0
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
    mask_shezhi = (label == fields[1])
    shezhi = np.zeros(img.shape)
    for i in range(3):
        shezhi[:,:,i] = mask_shezhi * img[:,:,i]
    # cv2.imwrite('shezhi.jpg',shezhi)

    mask_shetai = (label == fields[2])
    shetai = np.zeros(img.shape)
    for i in range(3):
        shetai[:,:,i] = mask_shetai * img[:,:,i]
    # cv2.imwrite('shetai.jpg',shetai)

    # cv2.imwrite('test.jpg', label)
    msk = mask_shetai*np.ones(mask_shetai.shape)*255
    return shetai, shezhi, msk

def getImageFromMask(imgpath, mskpath):
    '''
    img为原始图片，mask为单通道
    '''
    img = cv2.imread(imgpath)
    msk = cv2.imread(mskpath, 0)
    msk = msk > 200
    # print(np.sum(msk))
    # print(np.max(msk))
    # print(1)
    # for i in msk:
    #     for j in i:
    #         if j !=0:
    #             print(j)
    res = np.zeros(img.shape)
    for i in range(3):
        res[:,:,i] = msk * img[:,:,i]
    return res

# res = getImageFromMask('./test_data/0_raw.jpg', './test_data/0_mask.jpg')
res = cv2.imread('./test_data/tongueTop_seg.jpg')
shetai, shezhi,_ = cluster(res)
cv2.imwrite('shezhi.jpg', shezhi)
cv2.imwrite('shetai.jpg', shetai)