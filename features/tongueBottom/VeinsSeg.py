"""
作者：侯英妍
功能：舌下脉络语义分割
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
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset.features.tongueTop import tongueTopDataset

__all__ = ['viSeg']

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU()
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU()
        )
        # same: 2 * padding = dilation * (kernel_size - 1)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 32, 5, dilation=2, padding=4),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 32, 5, dilation=3, padding=6),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 5, dilation=5, padding=10),
            nn.ReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(32, 32, 5, dilation=7, padding=14),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU()
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        out = self.conv_0(x)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        out = self.conv_7(out)
        return out
    
class veinSeg():
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
        self.model =  FCN()
        
        if use_pretrained:
            # detect the pretrained path
            try:
                self.pretrained_path = glob('pretrained/features/tongueBottom/tongue_bottom_seg-*.pth')[0] if pretrained_path == "Default" else pretrained_path
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
            transforms.Resize([400, 300]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.is_train = is_train
        if self.is_train:
            # 定义损失函数
            self.loss_func = self.__loss_hinge
            # 创建数据集
            # TODO: 定义dataloader迭代器根据tongueTopDataloader类和标准化的数据集格式
            self.dataset = tongueTopDataset(dataset_path, self.transform)
            # batch_size = 64
            pass
    
    def __loss_hinge(self, y_ture, y_pred):
        hinge = torch.where(y_ture < 0.5, F.relu(y_pred), 8 * F.relu(-y_pred + 1))
        loss = torch.mean(hinge)
        return loss
    
    def __metrics(self, y_pred, y_true):
        """
        可以定义一些评价指标，准确率等等
        return {"Accuracy": ***, "F1_score": ***}
        """
        pass

    def do_train(self, epoch_num=300):
        # train & test are dataloader
        optimizer=optim.RMSprop(self.model.parameters(),lr=0.0005,alpha=0.9,eps=1e-06)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50, cooldown=0, min_lr=0, verbose=True)
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
            # Note that step should be called after validate()
            scheduler.step(test_result['test_loss'])

            print('Epoch: %d >> epoch train loss = %f, epoch test loss = %f , epoch F1-score = %s' \
                %(cur_epoch, train_loss / len(self.train_dataloader), test_result['test_loss'])) 

            if test_result['test_loss'] < best_loss:
                best_loss = test_result['test_loss']
                best_epo = cur_epoch
                torch.save(self.model.cpu().state_dict(), 'tongue_bottom_seg-fcn.pth')

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
        output = self.do_test(img)
        # TODO: 转化（得到指定分割模型的预测结果，并转换为灰度图像（0 / 1）输出）
        # output = y_pred.view(2,400,300)
        # output = output[0] - output[1]
        # output[output > 0] = 255
        # output[output <= 0] = 0
        # 输出结果
        ret = {
            "colorValue": [255, 255, 255],
            "imageData": output,
            "imageSize": [512,512]
        }
        return ret

if __name__ == "__main__":
    # 如果需要对模型进行训练，请在这里进行
    model = veinSeg(is_train=True, dataset_path="***", gpu_device_id=0)
    model.do_train(epoch_num=100)