from torch.utils.data import Dataset

__all__ = ['tongueBottomDataset']

class tongueBottomDataset(DataSet):
    def __init__(self, dataset_path, transform):
        self.transform = transform
        pass

    def __len__(self):
        pass

    def datapre(self):
        pass

    def __getitem__(self, item):
        # sample中可以增加其他标签信息
        # sample = {
        #     "image": self.transform(img),
        #     "tongue_label": tongue_label, # 舌体label
        #     "tooth_mask_label": tooth_mask_label, # 齿痕label
        #     "index": image_id
        # }
        # return sample
        pass

if __name__ == "__main__":
    pass