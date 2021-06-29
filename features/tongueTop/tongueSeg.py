"""
作者：常然
功能：舌体提取
"""
import os
import cv2
import base64
import pickle
import collections
import numpy as np
from sklearn.cluster import KMeans

from tqdm import tqdm
from glob import glob
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from data.dataset.features.tongueTop import tongueTopDataset

__all__ = ['tongueSeg']

class FCN32(nn.Module):
    def __init__(self, classnum = 2):
        super().__init__()
        self.pooling = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(inplace=True)
        self.conv0_1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv0_2 = nn.Conv2d(64, 64, 3, padding = 1)

        self.conv1_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv1_2 = nn.Conv2d(128, 128, 3, padding = 1)

        self.conv2_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv2_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv2_3 = nn.Conv2d(256, 256, 3, padding = 1)

        self.conv3_1 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv3_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv3_3 = nn.Conv2d(512, 512, 3, padding = 1)

        self.conv4_1 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding = 1)

        self.conv5 = nn.Conv2d(512, 4096, 7, padding = 3)
        self.drop = nn.Dropout2d()

        self.conv6 = nn.Conv2d(4096, 4096, 1)

        self.score_fr = nn.Conv2d(4096, classnum, 1)
        self.score_pool4 = nn.Conv2d(512, classnum, 1)

        self.upscore2 = nn.ConvTranspose2d(classnum, classnum, 4, stride=2,bias=False, padding=1)
        self.upscore16 = nn.ConvTranspose2d(classnum, classnum, 32, stride=16, bias=False, padding=8)
        self.upscore32 = nn.ConvTranspose2d(classnum, classnum, 64, stride=32, bias=False, padding=16)

    def forward(self, input):
        out = input
        # 1/2
        out = self.conv0_1(out)
        out = self.relu(out)
        out = self.pooling(out)
        # 1/4
        out = self.conv1_1(out)
        out = self.relu(out)
        out = self.pooling(out)
        # 1/8
        out = self.conv2_1(out)
        out = self.relu(out)
        out = self.pooling(out)
        # 1/16
        out = self.conv3_1(out)
        out = self.relu(out)
        out = self.relu(out)
        out = self.pooling(out)
        # 1/32
        out = self.conv4_1(out)
        out = self.relu(out)
        out = self.pooling(out)

        out = self.conv5(out)
        out = self.drop(out)
        out = self.conv6(out)
        out = self.drop(out)
        out = self.score_fr(out)

        out = self.upscore32(out)
        out = torch.sigmoid(out)
        return out
    
class tongueSeg():
    def __init__(self,
                is_train=False,
                dataset_path="",
                gpu_device_id=0,
                use_pretrained=True,
                pretrained_path="Default"):
        """
        is_train: 是否需要对模型进行重新训练
        dataset_path: 数据集路径
        gpu_device_id: 使用的gpu_id
        use_pretrained: 是否加载预训练模型
        pretrained_path: 预训练模型路径
        """
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() and gpu_device_id > 0 else 'cpu')
        self.model =  FCN32(classnum=2)
        
        if use_pretrained:
            # detect the pretrained path
            try:
                self.pretrained_path = glob('pretrained/features/tongueTop/tongue_seg-*.pt')[0] if pretrained_path == "Default" else pretrained_path
                assert os.path.exists(self.pretrained_path)
            except:
                raise ValueError("(default) pretrained_path is invalid!")
            try:
                # load pretrained parameters
                self.model.load_state_dict(torch.dict(self.pretrained_path))
            except:
                # load the whole model
                self.model = torch.load(self.pretrained_path)

        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.is_train = is_train
        if self.is_train:
            # 定义损失函数
            self.loss_func = nn.BCELoss()
            # 创建数据集
            # TODO: 定义dataloader迭代器根据tongueTopDataloader类和标准化的数据集格式
            self.dataset = tongueTopDataset(dataset_path, self.transform)
            pass
    
    def metrics(self, y_pred, y_true):
        """
        可以定义一些评价指标，准确率等等
        return {"Accuracy": ***, "F1_score": ***}
        """
        pass

    def do_train(self, epoch_num=50):
        # train & test are dataloader
        optimizer=optim.RMSprop(self.model.parameters(),lr=0.0001,alpha=0.9,eps=1e-06)
        best_loss = 1e8
        best_epoch = 0
        for cur_epoch in range(epoch_num):
            train_loss=0.0
            # 训练
            self.model.train()
            for data in tqdm(self.train_dataloader):
                inputs = data['inputs'].to(self.device)
                labels = data['labels'].to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                #print(output)
                loss = self.loss_func(output,labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # 测试
            test_result = self.do_test()

            print('Epoch: %d >> epoch train loss = %f, epoch test loss = %f , epoch F1-score = %s' \
                %(cur_epoch, train_loss / len(self.train_dataloader), test_result['test_loss'])) 

            if test_result['test_loss'] < best_loss:
                best_loss = test_result['test_loss']
                best_epo = cur_epoch
                torch.save(self.model.cpu().state_dict(), 'tongue_seg-fcn32.pth')

            if cur_epoch - best_epo > 20:
                print('Early Stop...')
                return

    def do_test(self, img=None):
        """
        img参数仅在self.is_train为True时有效
        """
        self.model.eval()
        if self.is_train:
            test_loss=0
            with torch.no_grad():
                for data in tqdm(self.test_dataloader):
                    inputs = data['inputs'].to(self.device)
                    labels = data['labels'].to(self.device)
                    # 输出
                    output = self.model(inputs)
                    # 计算结果
                    loss = self.loss_func(output,labels)
                    test_loss += loss.item()
                test_loss = test_loss / len(self.test_dataloader)
            return {"prediction": output, "test_loss": test_loss}
        else:
            img = self.transform(img).to(self.device)
            with torch.no_grad():
                output = self.model(img)
            return output.detach().cpu().numpy()

    def get(self, img):
        """
        function:
            舌体分割提取
        parameters:
            img: 舌部图片
        return:
            res: 结果数据，返回字典
                {
                    "colorValue": [255, 255, 255]
                    "imageData": img,
                    "imageSize": [100, 100]
                }
        """
        y_pred = self.do_test(img)
        # 转化（得到指定分割模型的预测结果，并转换为灰度图像（0 / 1）输出）
        output = y_pred.view(2,512,512)
        output = output[0] - output[1]
        output[output > 0] = 255
        output[output <= 0] = 0
        # 输出结果
        ret = {
            "colorValue": [255, 255, 255],
            "imageData": output,
            "imageSize": [512,512]
        }
        return ret

if __name__ == "__main__":
    # 如果需要对模型进行训练，请在这里进行
    model = tongueSeg(is_train=True, dataset_path="***", gpu_device_id=0)
    model.do_train(epoch_num=100)