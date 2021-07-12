import logging
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ['tongueBottomDataset']
logger = logging.getLogger()

class tongueBottomDataset(Dataset):
    def __init__(self, dataset_path, transform=transforms.ToTensor()):
        self.transform = transform
        if '舌下' in os.listdir(dataset_path):
            self.data_path = os.path.join(dataset_path, '舌下')
        else:
            logger.error('Dataset has no tongueBottom modality.')
            raise FileNotFoundError # TODO: define custom exception
        with open(os.path.join(self.data_path, 'labels.csv'), 'r') as f:
            self.labels = pd.read_csv(f)
        self.img_path = os.path.join(self.data_path, 'raw') # TODO: change 'raw' to 'processed'

    def __len__(self):
        return len(self.labels)

    # def datapre(self):
    #     pass

    def __getitem__(self, index):
        sample_id = int(self.labels.iloc[index]['sample_id'])
        image = Image.open(os.path.join(self.img_path, str(sample_id) + '.jpg'))
        image = self.transform(image)
        sample = {
            "sample_id": sample_id,
            "image": image,
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
    dataset = tongueBottomDataset(path)
    item = dataset.__getitem__(0)
    print(item)
