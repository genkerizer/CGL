import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data.distributed import DistributedSampler
from src.models_fusionv1.net import BaseModel
from src.loader.dataset import SimpleDataset
from src.losses.loss_func import *
from torch.utils.data import DataLoader


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Tester:
    
    def __init__(self, **kwargs):
        
        model_name = 'ViT-L/14'
        
        id_ = {'ViT-B/16': 'B16', 'ViT-B/32': 'B32', 'ViT-L/14': 'L14'}
        
        self.cfg = {
            'log_path': f'logs/test/{id_[model_name]}/20240822_fusionv1_6_384.txt',
            'logit_dir': 'statistic_tools/inputs/logits/20240822_fusionv1_6_384.json',
            'pretrain': 'checkpoints/L14/20240822_fusionv1_clip_6_384/best_eval_acc.pt',
            'return_logit': True,
            'save_logit_path': 'statistic_tools/inputs/logits/20240822_fusionv1_6_384.json',
            'batch_size': 24,
            'patch_size': 32,
            'img_size': [224, 224],
            'class_name': ["real", "synthetic"],
            'data_test_subset': ['DATASET/test/UniversalFakeDetect', 'DATASET/test/ForenSynths', \
                                'DATASET/test/DiffusionForensics_test_release']
            }
        
        self.return_logit = self.cfg['return_logit']
        self.cfg['patch_size'] = int(model_name.split('/')[-1])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BaseModel(model_name, self.cfg['class_name']).to(self.device)
        
        inputs = torch.randn((2, 3, 224, 224)).to(self.device)
        tokens = torch.randint(0, 100, size=(2, 77)).to(self.device)
        labels = torch.randint(0, 1, size=(2,)).to(self.device)
        self.model(inputs, tokens, labels)
        
        self.count_params()
        
        if self.cfg['pretrain'] is not None and os.path.exists(self.cfg['pretrain']):
            print("Load Pretrain")
            checkpoints = torch.load(self.cfg['pretrain'], map_location='cpu')
            print(checkpoints['lora_config'])
            self.model.cfg = checkpoints['lora_config']
            self.model.load_state_dict(checkpoints['model_state_dict'])
            lora_path = self.cfg['pretrain'].replace('.pt', '_lora.pt')
            self.model.load_lora_weight(lora_path)
            print('Loaded')
        
       
        self.model.to(self.device)
        self.model.eval()
        os.makedirs('/'.join(self.cfg['log_path'].split('/')[:-1]), exist_ok=True)
        
        
    def count_params(self):
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Number of trainable parameters: {num_trainable_params}")
            

    def eval(self, test_dataloader, return_logit=True):
        
        if return_logit:
            real_logit = []
            fake_logit = []
        
        y_true, y_pred = [], []
    
        for samples in tqdm(test_dataloader):
            
            imgs, labels = samples[:2]
            imgs = imgs.to(self.device)

            outputs = self.model(imgs)
            outputs = outputs.flatten()
            
            real_logit.extend(outputs[labels==0].tolist())
            fake_logit.extend(outputs[labels==1].tolist())
            
            
            outputs = torch.sigmoid(outputs)
            
            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())
            
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        if return_logit:
            return acc, ap, r_acc, f_acc, real_logit, fake_logit
        
        return acc, ap, r_acc, f_acc
            
    
    def train(self):
        
        log_w = open(self.cfg['log_path'], 'w')
        log_w.writelines(f'============================TESTING============================\n')
        
        with torch.no_grad():
            outputs_logits = {}
            
            for datatest_type in self.cfg['data_test_subset']:
                datatest_type_name = datatest_type.split('/')[-1]
                
                sub_test_dir_list = os.listdir(datatest_type)
                
                self.test_dataloader_list = []
                for sub_test_dir_name in sub_test_dir_list:
                    data_test_dir = os.path.join(datatest_type, sub_test_dir_name)
                    
                    test_dataset = SimpleDataset(self.cfg, [data_test_dir], training=False)
                    test_dataloader = DataLoader(test_dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=2)
                    self.test_dataloader_list.append([sub_test_dir_name, test_dataloader])
            
                
                total_samples = 0
                m_acc, m_ap = [], []
                
                myTable = PrettyTable(["Data name", "Num sample", "Acc", "AP", "R_acc", "F_acc"])
                
                for dataname, test_dataloader in self.test_dataloader_list:
                    
                    temple_show_myTable = PrettyTable(["Data name", "Num sample", "Acc", "AP", "R_acc", "F_acc"])
                    
                    if self.return_logit:
                        acc, ap, r_acc, f_acc, real_logit, fake_logit = self.eval(test_dataloader, return_logit=True)
                        outputs_logits[dataname] = {'real': real_logit, 'fake': fake_logit}
                    
                    
                    else:
                        acc, ap, r_acc, f_acc = self.eval(test_dataloader, return_logit=True)
                    
                    
                    n_samples = len(test_dataloader.dataset)
                    total_samples += n_samples
                    
                    myTable.add_row([dataname, str(n_samples), str(round(acc, 4)), \
                        str(round(ap, 4)), str(round(r_acc, 4)), str(round(f_acc, 4))])
                    
                    temple_show_myTable.add_row([dataname, str(n_samples), str(round(acc, 4)), \
                        str(round(ap, 4)), str(round(r_acc, 4)), str(round(f_acc, 4))])
                    
                    print('\n')
                    print(temple_show_myTable)
                    
                    m_acc.append(acc)
                    m_ap.append(ap)
                    
                print('\n')
                print(myTable)

                log_w.writelines('\n')
                log_w.writelines("=" * 30)
                log_w.writelines('\n')
                log_w.writelines(str(myTable))
                log_w.flush()
                
                    
                m_acc = np.mean(m_acc)
                m_ap = np.mean(m_ap)
                
                myTable = PrettyTable(["Data Test Name", "Num sample", "Mean Acc", "Mean AP"])
                    
                myTable.add_row([datatest_type_name, str(total_samples), \
                    str(np.round(m_acc, 4)), str(np.round(m_ap, 4))])
                
                print('\n')
                print(myTable)
                
                log_w.writelines('\n')
                log_w.writelines(str(myTable))
                log_w.writelines('\n')
                log_w.flush()
                            
            if self.return_logit:
                with open(self.cfg['save_logit_path'], 'w') as f:
                    json.dump(outputs_logits, f, indent=2)               
            
        log_w.close()
                
    
if __name__ == '__main__':
    seed_torch(100)
    trainer = Tester()
    trainer.train()