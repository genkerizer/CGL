import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, accuracy_score


from src.models.net import BaseModel
from src.loader.dataset import SimpleDataset
from torch.utils.data import DataLoader
from src.losses.loss_func import *


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


class Trainer:
    
    def __init__(self, **kwargs):
        
        model_name = 'ViT-L/14'
        
        id_ = {'ViT-B/16': 'B16', 'ViT-B/32': 'B32', 'ViT-L/14': 'L14'}
    
        self.cfg = {
            'epoch': 25,
            'lr': 1e-4,
            'checkpoint_dir': f'checkpoints/{id_[model_name]}/20240822_fusionv1_stop_grad',
            'log_path': f'logs/train/{id_[model_name]}/20240822_fusionv1_stop_grad.txt',
            'test_log_path': f'logs/test/{id_[model_name]}/20240822_fusionv1_stop_grad.txt',
            'pretrain': None, 
            'patch_size': 32,
            'iter_eval': 400,
            'train_batch_size': 16,
            'test_batch_size': 24,
            'img_size': [224, 224],
            'class_name': ["real", "synthetic"],
            'caption_train': 'CAPTIONS/BLIP/CAPTION_full',
            'data_train': ['BACKUP/train/car', 'BACKUP/train/cat', 'BACKUP/train/chair', 'BACKUP/train/horse'],
            'data_test': ['DATASET/test/Diffusion1kStep', 'DATASET/test/DiffusionForensics_test_release', 'DATASET/test/ForenSynths', 'DATASET/test/GANGen-Detection', 'DATASET/test/UniversalFakeDetect'],
            'data_valid': ['DATASET/test/UniversalFakeDetect/dalle', 'DATASET/test/GANGen-Detection/AttGAN', 'DATASET/test/ForenSynths/biggan', 'DATASET/test/ForenSynths/deepfake']
        
        }
        
        self.cfg['patch_size'] = int(model_name.split('/')[-1])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = BaseModel(model_name, self.cfg['class_name'])
        self.model.to(self.device)
        
        inputs = torch.randn((2, 3, 224, 224)).to(self.device)
        tokens = torch.randint(0, 100, size=(2, 77)).to(self.device)
        labels = torch.randint(0, 1, size=(2,)).to(self.device)
        self.model(inputs, tokens, labels)
        self.freeze_layers()
    
        if self.cfg['pretrain'] is not None and os.path.exists(self.cfg['pretrain']):
            print("Load Pretrain")
            self.model.load_state_dict(torch.load(self.cfg['pretrain'], map_location='cpu')['model_state_dict'])
            lora_path = self.cfg['pretrain'].replace('.pt', '_lora.pt')
            self.model.load_lora_weight(lora_path)
            print("Loaded")
        
        self.loss = CrossEntropyFusionLoss()
        n_iter = 500
        shots = 16
        
        total_iters = n_iter * shots
        self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=1e-2, betas=(0.9, 0.999), lr=self.cfg['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_iters, eta_min=1e-6)
        
        train_dataset = SimpleDataset(self.cfg, self.cfg['data_train'], self.cfg['caption_train'], training=True)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.cfg['train_batch_size'], shuffle=True, num_workers=4)
            
        self.valid_dataloader_list = []
        for data_test_dir in self.cfg['data_valid']:
            name = data_test_dir.split('/')[-1]
            
            valid_dataset = SimpleDataset(self.cfg, [data_test_dir], training=False)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.cfg['test_batch_size'], shuffle=False, num_workers=4)
            self.valid_dataloader_list.append([name, valid_dataloader])
            
        os.makedirs(self.cfg['checkpoint_dir'], exist_ok=True)
        os.makedirs('/'.join(self.cfg['log_path'].split('/')[:-1]), exist_ok=True)
        self.iter_eval = self.cfg['iter_eval']
        
        
    def freeze_layers(self):
        UPDATE_NAMES = ['lora_', 'classifcation_head']
        
        for name, param in self.model.named_parameters():
            update = False
            for name_to_update in UPDATE_NAMES:
                if name_to_update in name:
                    update = True
                    break
    
            param.requires_grad_(update)
                
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")
        
        
    def eval(self, test_dataloader):
        
        y_true, y_pred = [], []
    
        for samples in tqdm(test_dataloader):
            
            imgs, labels = samples[:2]
            imgs = imgs.to(self.device)

            outputs = self.model(imgs)
            outputs = torch.sigmoid(outputs)
            
            y_pred.extend(outputs.flatten().tolist())
            y_true.extend(labels.flatten().tolist())
            
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        return acc, ap, r_acc, f_acc
    
    
    def train(self):
        
        self.model.train()
        self.model.to(self.device)
        scaler = torch.cuda.amp.GradScaler()
        
        step = 0
        best_acc = 0.0
        num_epoch = self.cfg['epoch'] + 1
        
        log_w = open(self.cfg['log_path'], 'w')
        
        for epoch in range(1, num_epoch):
            
            self.train_dataloader.dataset.shuffle()
            
            loop = tqdm(self.train_dataloader)
            loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
            
            for samples in loop:
                
                imgs, labels, tokenize = samples
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                tokenize = tokenize.to(self.device)
                
                predicts = self.model(imgs, text=tokenize, label=labels)
                
                total_loss = self.loss(predicts, labels)
                
                if torch.isnan(total_loss):
                    print(total_loss)
                    continue
                
                loop.set_postfix(loss=total_loss.item())
                self.optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                
                step += 1
                
                if step % self.iter_eval == 0:
                    torch.save({'model_state_dict': self.model.state_dict(), 'lora_config': self.model.cfg}, 
                                       os.path.join(self.cfg['checkpoint_dir'], 'iter.pt'))
                    self.model.save_lora_weight(os.path.join(self.cfg['checkpoint_dir'], 'iter_lora.pt'))
                    
                    log_w.writelines(f'============================{step}============================\n')
                
                    with torch.no_grad():
                        self.model.eval()
                        
                        total_samples = 0
                        c_acc, c_ap = [], []
                        
                        myTable = PrettyTable(["Epoch", "Iter", "Data name", "Num sample", "Acc", "AP", "R_acc", "F_acc"])
                        
                        for dataname, valid_dataloader in self.valid_dataloader_list:
                            acc, ap, r_acc, f_acc = self.eval(valid_dataloader)
                            n_samples = len(valid_dataloader.dataset)
                            total_samples += n_samples
                            
                            myTable.add_row([str(epoch), str(step), dataname, str(n_samples), str(round(acc, 4)), \
                                str(round(ap, 4)), str(round(r_acc, 4)), str(round(f_acc, 4))])
                            
                            print(dataname, ":\t", acc)
                            
                            c_acc.append(acc)
                            c_ap.append(ap)
            
                            
                        print('\n')
                        print(myTable)
                        
                        log_w.writelines('\n')
                        log_w.writelines(str(myTable))
                        log_w.flush()
                            
                        c_acc = np.mean(c_acc)
                        c_ap = np.mean(c_ap)
                        
                        if best_acc < c_acc:
                            best_acc = c_acc
                            self.model.train()
                            torch.save({'model_state_dict':self.model.state_dict(), 'lora_config': self.model.cfg}, 
                                       os.path.join(self.cfg['checkpoint_dir'], 'best_eval_acc.pt'))
                            self.model.save_lora_weight(os.path.join(self.cfg['checkpoint_dir'], 'best_eval_acc_lora.pt'))
                            
                        myTable = PrettyTable(["Epoch", "Iter", "Num sample", "Best Acc", "Curent Acc", "Curent AP"])
                            
                        myTable.add_row([str(epoch), str(step), str(total_samples), str(np.round(best_acc, 4)), \
                            str(np.round(c_acc, 4)), str(np.round(c_ap, 4))])
                        
                        print('\n')
                        print(myTable)
                        
                        log_w.writelines('\n')
                        log_w.writelines(str(myTable))
                        log_w.writelines('\n')
                        log_w.flush()
                        
                    self.model.train()
                            
            torch.save({'model_state_dict':self.model.state_dict(), 'lora_config': self.model.cfg}, 
                       os.path.join(self.cfg['checkpoint_dir'], f'epoch_{epoch}.pt'))
            self.model.save_lora_weight(os.path.join(self.cfg['checkpoint_dir'], f'epoch_{epoch}_lora.pt'))
            
        log_w.close()
             
    
if __name__ == '__main__':
    seed_torch(100)
    trainer = Trainer()
    trainer.train()