import logging
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['tongueBottomDataset']
logger = logging.getLogger()

class tongueTopDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.transform = transform
        if '舌上' in os.listdir(dataset_path):
            self.data_path = os.path.join(dataset_path, '舌上')
        else:
            logger.error('Dataset has no tongueTop modality.')
            raise FileNotFoundError # TODO: define custom exception
        with open(os.path.join(self.data_path, 'labels.csv'), 'r') as f:
            self.labels = pd.read_csv(f)
        self.img_path = os.path.join(self.data_path, 'processed')
        if not os.path.exists(self.img_path):
            pass # TODO: 调用数据预处理
        self.raw_path = os.path.join(self.data_path, 'raw')

    def __len__(self):
        return len(self.labels)

    # def datapre(self):
    #     pass

    def __getitem__(self, index):
        sample_id = int(self.labels.iloc[index]['sample_id'])
        image = Image.open(os.path.join(self.img_path, str(sample_id) + '.jpg'))
        raw_img = Image.open(os.path.join(self.raw_path, str(sample_id) + '.jpg'))
        if self.transform is not None:
            image = self.transform(image)
            raw_img = self.transform(raw_img)
        sample = {
            "sample_id": sample_id,
            "image": image,
            "raw_img": raw_img, # Used for tooth mark detection
            "coating_color": self.labels.iloc[index]['coating_color'],
            "coating_L_value": self.labels.iloc[index]['coating_L_value'],
            "coating_A_value": self.labels.iloc[index]['coating_A_value'],
            "coating_B_value": self.labels.iloc[index]['coating_B_value'],
            "tongue_width": self.labels.iloc[index]['tongue_width'],
            "tongue_thickness": self.labels.iloc[index]['tongue_thickness'],
            "tongue_color": self.labels.iloc[index]['tongue_color'],
            "tongue_L_value": self.labels.iloc[index]['tongue_L_value'],
            "tongue_A_value": self.labels.iloc[index]['tongue_A_value'],
            "tongue_B_value": self.labels.iloc[index]['tongue_B_value'],
            "track": self.labels.iloc[index]['track'],
            "track_index": self.labels.iloc[index]['track_index'],
            "tooth_mark": self.labels.iloc[index]['tooth_mark'],
        }
        return sample

if __name__ == "__main__":
    path = '/home/sharing/disk3/Datasets/TCM-Datasets/多模态数据'
    dataset = tongueTopDataset(path)
    item = dataset.__getitem__(0)
    print(item)
