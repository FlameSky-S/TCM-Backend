"""
作者：李华东
功能：脸部语义区域分割提取
"""
import os
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from data.dataset.features.face import faceDataset

# __all__ = ['AreaSeg']

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        self.n_classes = num_classes

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

class AreaSeg():
    def __init__(self,
                # is_train=False,
                dataset_path="",
                gpu_device_id=0,
                use_pretrained=True,
                pretrained_path="Default",
                num_classes=15):
        """
        is_train: 是否需要对模型进行重新训练
        dataset_path: 数据集路径
        gpu_device_id: 使用的gpu_id
        use_pretrained: 是否加载预训练模型
        pretrained_path: 预训练模型路径
        """
        self.device = torch.device(f'cuda:{gpu_device_id}' if torch.cuda.is_available() and gpu_device_id > 0 else 'cpu')
        self.model = NestedUNet(num_classes=15, input_channels=3)
        
        if use_pretrained:
            try:
                self.pretrained_path = glob.glob('pretrained/features/face/area_seg-*.pth')[0] if pretrained_path == "Default" else pretrained_path
                assert os.path.exists(self.pretrained_path)
            except:
                raise ValueError("Invalid pretrained_path.")
            try:
                # load pretrained parameters
                self.model.load_state_dict(torch.load(self.pretrained_path))
            except:
                # load the whole model
                self.model = torch.load(self.pretrained_path)

        self.model.to(self.device)
        # 可考虑在这里加一些数据增强方法(要同时考虑标签的转换)
        self.transform = transforms.Compose([
            # transforms.Resize([512, 512]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        # self.is_train = is_train
        # self.num_classes = num_classes
        # if self.is_train:
            # 定义损失函数
            # self.loss_func = DiceLoss()
            # 创建数据集
            # TODO: 定义dataloader迭代器根据tongueTopDataloader类和标准化的数据集格式
            # self.dataset = faceDataset(dataset_path, self.transform)
            # batch_size = 64
            # pass
    
    def __metrics(self, y_pred, y_true):
        """
        可以定义一些评价指标，准确率等等
        return {"Accuracy": ***, "F1_score": ***}
        """
        pass

    def do_train(self, epoch_num=300):
        # train & test are dataloader
        optimizer=optim.RMSprop(self.model.parameters(), lr=0.0005, weight_decay=1e-8, momentum=0.9)
        best_loss = 1e8
        best_epoch = 0
        for cur_epoch in range(epoch_num):
            train_loss=0
            # 训练
            self.model.train()
            for data in tqdm(self.train_dataloader):
                inputs = data['inputs'].to(self.device)
                labels = data['labels'].to(self.device)
                batch_size = inputs.size(0)

                optimizer.zero_grad()
                masks_pred = self.model(inputs)

                size_length = masks_pred.size()[2] * masks_pred.size()[3]
                mask_p = masks_pred.view(batch_size, self.num_classes, size_length)
                mask_t = labels.view(batch_size, self.num_classes, size_length)
                #print(output)
                loss = self.loss_func.forward(mask_p.float(), mask_t.float())
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                optimizer.step()
                train_loss += loss.item()
            # 测试
            test_result = self.do_test()

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
        img为PIL image
        """
        self.model.eval()
        # if self.is_train:
        #     test_loss=0
        #     with torch.no_grad():
        #         for data in tqdm(self.test_dataloader):
        #             inputs = data['inputs'].to(self.device)
        #             labels = data['labels'].to(self.device)
        #             batch_size = inputs.size(0)
        #             # 输出
        #             masks_pred = self.model(inputs)

        #             size_length = masks_pred.size()[2] * masks_pred.size()[3]
        #             mask_p = masks_pred.view(batch_size, self.num_classes, size_length)
        #             mask_t = labels.view(batch_size, self.num_classes, size_length)
        #             #print(output)
        #             loss = self.loss_func.forward(mask_p.float(), mask_t.float())
        #             test_loss += loss.item()
        #         test_loss = test_loss / len(self.test_dataloader)
        #     return {"prediction": masks_pred, "test_loss": test_loss}
        # else:
        img = self.transform(img).to(self.device)
        # img = img.to(self.device)
        img = img.unsqueeze(0)
        with torch.no_grad():
            output = self.model(img).detach().cpu()
        mask_pred = torch.squeeze(torch.max(output, 1)[1]).numpy().astype(np.uint8)
        return mask_pred

    def get(self, img):
        """
        function:
            处理单张人脸图片
        parameters:
            src_path: 人脸图片路径（无其他背景的纯人脸图片）
            dst_path: 结果保存路径，若为空，则不保存（保存成相应的json文件）
        return:
            res: 结果数据，返回字典
                {
                    "loc1":{
                        "labelname": "额头左侧",
                        "colorValue": [255, 255, 255]
                    },
                    "loc2":{
                        "labelname": "额头右侧",
                        "colorValue": [155, 155, 155]
                    },
                    "imageData": "base64编码, 见utils.tools.img_to_txt",
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
        # res: 结果数据，返回字典
        # {
        #     "loc1":{
        #         "labelname": "额头左侧",
        #         "colorValue": [255, 255, 255]
        #     },
        #     "loc2":{
        #         "labelname": "额头右侧",
        #         "colorValue": [155, 155, 155]
        #     },
        #     "imageData": "base64编码, 见utils.tools.img_to_txt",
        #     "imageSize": [100, 100]
        # }
        return