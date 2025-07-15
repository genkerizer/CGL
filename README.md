# **[Caption Guided Learning with Lora for Generalizable Deepfake Detection]**

<p align="center">
<img src="https://img.shields.io/aur/last-modified/google-chrome">
<img src="https://img.shields.io/badge/Author-Y--Hop.Nguyen-red"> 
</p>

This repository is an official implementation of the S 2024 paper "Caption Guided Learning with Lora for Generalizable Deepfake Detection".


# Introduction

![CGL architecutures: Training stage and Inference Stage ](assets/overall_architecture.svg)

# Dataset

## Getting the data

Download dataset from [CNNDetection CVPR2020 (Table1 results)](https://github.com/peterwang512/CNNDetection), [GANGen-Detection (Table2 results)](https://github.com/chuangchuangtan/GANGen-Detection) ([googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)), [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) ([googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=drive_link)), [DIRE 2023ICCV](https://github.com/ZhendongWang6/DIRE) ([googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)), Diffusion1kStep [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing).




# Training

Setup enviroment by runing command: 
```
pip install -r requirements.txt
python3 train.py
```


# Evaluate

# References

