from torch.utils.data import Dataset
import os
import logging
import pandas as pd
from PIL import Image
from torchvision import transforms

__all__ = ['faceDataset']
logger = logging.getLogger()

class faceDataset(Dataset):
    def __init__(self, dataset_path, transform=transforms.ToTensor()):
        self.transform = transform
        if '面诊' in os.listdir(dataset_path):
            self.data_path = os.path.join(dataset_path, '面诊')
        else:
            logger.error('Dataset has no face modality.')
        with open(os.path.join(self.data_path, 'labels.csv'), 'r') as f:
            self.labels = pd.read_csv(f)
        self.img_path = os.path.join(self.data_path, 'raw') # TODO: change 'raw' to 'processed'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample_id = int(self.labels.iloc[index]['sample_id'])
        image = Image.open(os.path.join(self.img_path, str(sample_id) + '.jpg'))
        image = self.transform(image)
        sample = {
            "sample_id": sample_id,
            "image": image,
            "face_color": self.labels.iloc[index]['face_color'],
            "face_L_value": self.labels.iloc[index]['face_L_value'],
            "face_A_value": self.labels.iloc[index]['face_A_value'],
            "face_B_value": self.labels.iloc[index]['face_B_value'],
            "lip_color": self.labels.iloc[index]['lip_color'],
            "lip_L_value": self.labels.iloc[index]['lip_L_value'],
            "lip_A_value": self.labels.iloc[index]['lip_A_value'],
            "lip_B_value": self.labels.iloc[index]['lip_B_value'],
            "face_light_result": self.labels.iloc[index]['face_light_result'],
            "face_light_index": self.labels.iloc[index]['face_light_index'],
            "face_less_light_index": self.labels.iloc[index]['face_less_light_index'],
            "face_no_light_index": self.labels.iloc[index]['face_no_light_index'],
        }
        return sample

if __name__ == "__main__":
    path = '/home/sharing/disk3/Datasets/TCM-Datasets/多模态数据'
    dataset = faceDataset(path)
    item = dataset.__getitem__(0)
    print(item)
