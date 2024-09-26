
import os
import cv2
import json
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from .utils import *
from .aug import DataAug
from src.models.clip.clip import tokenize as Tokenize_func


class SimpleDataset(Dataset):
    
    def __init__(self, cfg, data_dir_list, caption=None, noise_data=None, training=False, **kwargs):
        super().__init__()
        
        
        self.training = training
        self.img_size = cfg['img_size']
        self.data_dir_list = data_dir_list
        self.caption = caption
        self.noise_data = noise_data
        self.patch_size = cfg['patch_size']
        self.class_name = cfg['class_name']
        self.num_patch = (self.img_size[0] // self.patch_size) ** 2
        
        self.aug = DataAug(self.patch_size)
        
        
        self.prepare_dataset_training()
        self.num_sample = self.dataset.shape[0]
        self.real_num_sample = self.real_dataset.shape[0]
        self.fake_num_sample = self.fake_dataset.shape[0]
        
        self.transform = self._transform(self.img_size)

        
    def prepare_dataset_training(self):
        
        dataset = []
        real_dataset = []
        fake_dataset = []
     
        
        for data_dir in self.data_dir_list:
            
            sub_dir = os.listdir(data_dir)
            
            if '0_real' and '1_fake' in sub_dir:
                
                if self.caption is not None:
                    name = data_dir.split('/')[-1]
                    real_caption_path = os.path.join(self.caption, name, '0_real.json')
                    fake_caption_path = os.path.join(self.caption, name, '1_fake.json')
                    real_caption = json.load(open(real_caption_path))
                    fake_caption = json.load(open(fake_caption_path))
                
                real_img_paths = glob.glob(os.path.join(data_dir, '0_real/*.*'))
                fake_img_paths = glob.glob(os.path.join(data_dir, '1_fake/*.*'))

                real_label = [0] * len(real_img_paths)
                fake_label = [1] * len(fake_img_paths)
                
                if self.caption is not None:
                
                    for path_, cls_ in zip(real_img_paths + fake_img_paths, real_label + fake_label):
                        caption_text = ''
                        img_name = path_.split('/')[-1]
                        if cls_ == 0:
                            caption_text = real_caption[img_name]
                            real_dataset.append([path_, cls_, caption_text])
                        else:
                            caption_text = fake_caption[img_name]
                            fake_dataset.append([path_, cls_, caption_text])
                        
                        dataset.append([path_, cls_, caption_text])
                else:
                    
                    for path_, cls_ in zip(real_img_paths + fake_img_paths, real_label + fake_label):
                        if cls_ == 0:
                            real_dataset.append([path_, cls_, ''])
                        else:
                            fake_dataset.append([path_, cls_, ''])
                        dataset.append([path_, cls_, ''])
                    
            else:
                data_type_list = os.listdir(data_dir)
                for data_type in data_type_list:
                    real_img_paths = glob.glob(os.path.join(data_dir, data_type, '0_real/*.*'))
                    fake_img_paths = glob.glob(os.path.join(data_dir, data_type, '1_fake/*.*'))
                    
                    real_label = [0] * len(real_img_paths)
                    fake_label = [1] * len(fake_img_paths)
                
                    for path_, cls_ in zip(real_img_paths + fake_img_paths, real_label + fake_label):
                        dataset.append([path_, cls_, ''])
                        if cls_ == 0:
                            real_dataset.append([path_, cls_, ''])
                        else:
                            fake_dataset.append([path_, cls_, ''])
                
        self.dataset = np.asarray(dataset)
        np.random.shuffle(self.dataset)
    
        self.real_dataset = np.asarray(real_dataset)
        self.fake_dataset = np.asarray(fake_dataset)
        np.random.shuffle(self.real_dataset)
        np.random.shuffle(self.fake_dataset)
        return None
    
    
    def shuffle(self):
        np.random.shuffle(self.dataset)
            
    
    def __len__(self):
        return self.dataset.shape[0]
    
    
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")
    
    
    def pre_process(self, img, label):
        img = np.array(img.convert('RGB'))
        
        if self.training:
            img = crop_with_size(img, self.img_size)
            # img = self.aug(img, None)
        else:
            img = center_crop_with_size(img, self.img_size)
            
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label
        
    
    def _transform(self, n_px):
        return Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        
    def __getitem__(self, index):
        img_path, label, caption = self.dataset[index]
        img = Image.open(img_path)
        
        label = int(label)
        caption = 'a photo is ' + self.class_name[label] + ' and ' + caption
        img, label = self.pre_process(img, label)
        label = torch.tensor(label, dtype=torch.float32)
        
        if not self.training:
            return img, label
        
        tokenize = Tokenize_func([caption])[0]
        return img, label, tokenize
    
    
if __name__ == '__main__':
    cfg = {
            'epoch': 10,
            'checkpoint_dir': 'checkpoints/20240704',
            'log_path': 'logs/20240704.txt',
            'pretrain': None,
            'iter_eval': 2000,
            'batch_size': 32,
            'img_size': [256, 256],
            'data_train': ['DATASET/train/car', 'DATASET/train/cat', 'DATASET/train/chair', 'DATASET/train/horse'],
            'data_test': ['DATASET/test/ForenSynths/progan', 'DATASET/test/ForenSynths/deepfake']
        }
    
    dataset = SimpleDataset(cfg, ['DATASET/test/GANGen-Detection/AttGAN'], True)
    from tqdm import tqdm
    for i in tqdm(range(0, len(dataset))):
        outputs = dataset[i]
        # print(outputs[0].shape)
        