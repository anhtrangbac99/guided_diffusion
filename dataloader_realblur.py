from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler
import os
from PIL import Image
class RealBlur(Dataset):
    def __init__(self,data_path):
        scenes = os.listdir(data_path)
        self.path_images = []
        self.path_gts = []
        for scence in scenes:
            path = os.path.join(data_path,scence)
            blur_path = os.path.join(path,'blur')
            gt_path = os.path.join(path,'gt')
            images = os.listdir(blur_path)
            gt = os.listdir(gt_path)
            for idx,_ in enumerate(images):
                self.path_images.append(os.path.join(blur_path,images[idx]))
                self.path_gts.append(os.path.join(gt_path,gt[idx]))
        self.trans = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    def __len__(self):
        return len(self.path_images)

    def __getitem__(self,idx):
        image = Image.open(self.path_images[idx])
        gt = Image.open(self.path_gts[idx])
        image = self.trans(image)
        gt = self.trans(gt)
        return image,gt


def load_data(batchsize:int, numworkers:int, data_path:str = None) -> tuple[DataLoader, DistributedSampler]:
    data_train = RealBlur(data_path)
    # sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers
                        # sampler = sampler,
                        # drop_last = True
                    )
    return trainloader

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
