# **[Decoding Deepfakes: Caption Guided Learning for Robust Deepfake Detection]**

<p align="center">
<img src="https://img.shields.io/aur/last-modified/google-chrome">
<img src="https://img.shields.io/badge/Author-Y--Hop.Nguyen-red"> 
</p>

This repository is an official implementation of SOICT 2024 paper "Decoding Deepfakes: Caption Guided Learning for Robust Deepfake Detection".


# Introduction

![CGL architecutures: Training stage and Inference Stage ](assets/overall_architecture.svg)

# Dataset

## Getting the data

Download dataset from [CNNDetection CVPR2020 (Table1 results)](https://github.com/peterwang512/CNNDetection), [GANGen-Detection (Table2 results)](https://github.com/chuangchuangtan/GANGen-Detection) ([googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing)), [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) ([googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=drive_link)), [DIRE 2023ICCV](https://github.com/ZhendongWang6/DIRE) ([googledrive](https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf?usp=sharing)), Diffusion1kStep [googledrive](https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq?usp=sharing).

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/genkerizer/CGL.git
   cd CGL
   ```

2. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

## How to training

# Training

Setup enviroment by runing command: 
```
pip install -r requirements.txt
python3 train.py
```
# Evaluate



# References

## License

**CGL** is published under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License.

You may use, share, and adapt this work **for non-commercial research and academic purposes only**.  
**Commercial use is strictly prohibited.**

🔗 [CC BY-NC 4.0 Legal Code](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Citation
If you use this code or its concepts in your research, please cite the following publication:

**Minimalist Preprocessing Approach for Image Synthesis Detection**  
Hoai-Danh Vo, Trung-Nghia Le  
*Information and Communication Technology (SOICT 2024), Communications in Computer and Information Science, vol 2350, Springer, Singapore, 2025.*  
[https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8](https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8)

```bibtex
@inproceedings{Vo2025MinimalistPA,
  author={Vo, Hoai-Danh and Le, Trung-Nghia},
  booktitle={Information and Communication Technology},
  doi={10.1007/978-981-96-4282-3_8},
  editor={Buntine, Wray and Fjeld, Morten and Tran, Truyen and Tran, Minh-Triet and Huynh Thi Thanh, Binh and Miyoshi, Takumi},
  isbn={978-981-96-4281-6},
  pages={88-99},
  publisher={Springer Nature Singapore},
  title={Minimalist Preprocessing Approach for Image Synthesis Detection},
  year={2025}
}
```
[Download BibTeX file](https://citation-needed.springer.com/v2/references/10.1007/978-981-96-4282-3_8?format=bibtex&flavour=citation)

---

## Related Publication

This section provides additional details about the academic paper that describes the core methodology of this project.

**Decoding Deepfakes: Caption Guided Learning for Robust Deepfake Detection**  
Y-Hop Nguyen, Trung-Nghia Le  
*Information and Communication Technology (SOICT 2024), Communications in Computer and Information Science, vol 2350, Springer, Singapore, 2025.*  
[https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8](https://link.springer.com/chapter/10.1007/978-981-96-4282-3_8)

**Abstract:**
> The rapid development of generative image models has raised concerns about misuse, especially in journalism and media. Therefore, developing tools for detecting fake images is essential. However, many current methods focus on short-term gains and lack long-term adaptability. This paper focuses on detecting deepfakes across various types of image data, such as faces, landscapes, objects, and scenes, using the visual-language CLIP model. Although CLIP has shown potential in deepfake detection, it has yet to clarify why it performs effectively in this task. Our analysis shows that CLIP’s combination of image features enhances the model’s generalization capability. By extracting image features trained for the deepfake detection task and generating captions through a text-decoding model, we demonstrate its effectiveness. Based on these findings, we introduce a novel method that enables the learning of forgery features and semantic features to improve generalization in
image forgery detection.

**Conclusion:**
> We present a novel Caption Guided Learning (CGL) method for generalizable image detection, incorporating three modules with CLIP to enhance feature extraction for deepfake detection. Extensive experiments on GAN and Diffusion model datasets show that CGL achieves state-of-the-art performance, highlighting its strong generalization capability. Additionally, the simplicity and flexibility of our approach may inspire further advancements in deepfake detection using frozen pre-trained models.


