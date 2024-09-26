
import cv2
import torch
import random
import numpy as np

def normalize(img):
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32) / 255.0
    return img

def text_encoder(label):
    token = 'Fake' if label else 'Real'
    return token

def crop_with_size(img, size=[256, 256]):
    
    h, w = img.shape[:2]
    if h == size[0] and w == size[1]:
        return img
    
    if min(h, w) >= max(size):
        start_x = random.randint(-1, w-size[1])
        start_y = random.randint(-1, h-size[0])
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        
        img = img[start_y:start_y + size[0], start_x:start_x + size[1]]
        return img
        
    return cv2.resize(img, size)
    
    
    
def center_crop_with_size(img, size=[256, 256]):
    
    h, w = img.shape[:2]
    
    if h == size[0] and w == size[1]:
        return img
        
    if min(h, w) >= max(size):
        center_x = w // 2
        center_y = h // 2
        start_x = max(0, center_x - size[1]//2)
        start_y = max(0, center_y - size[0]//2)
        img = img[start_y:start_y + size[0], start_x:start_x + size[1]]
        return img
        
    return cv2.resize(img, size)
    

if __name__ == '__main__':
    pass


