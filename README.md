# Frequency-Aware Masked-Attention for Infrared Small Target Detection

Pytorch implementation of our "FM-Net: Frequency-Aware Masked-Attention for Infrared Small Target Detection (Remote Sensing 2025)". [[Paper]](https://www.mdpi.com/2072-4292/17/13/2264)


## Requirements
- Python 3.12
- torch==2.3.0, torchvision==0.18.0, torchviz=0.0.3, Cuda 12.1
- einops==0.8.1, ml_collections==1.1.0
- pytorch-wavelets==1.3.0
- torch-dct==0.1.6
- openmim==0.3.9, mmcv==2.2.0
- timm==1.0.15
<br><br>

## Datasets
* SIRST &nbsp; [[download dir]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* IRSTD-1K &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
* NUDT-SIRST &nbsp; [[download dir]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9864119)

* **The organization of our dataset is as follows:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    ├── ...  
  ```
<be>

## Train
* **Run **`train.py`** to perform network training. Example for training [model_name] on [dataset_name] datasets:**
  ```
  python train.py --model_names FM-Net --dataset_names IRSTD-1K
  ```
* **Checkpoints and Logs will be saved to **`./log/`**:**
  ```
  ├──./log/
  │    ├── [dataset_name]
  │    │    ├── [model_name]_xxxxx.pth.tar
  ```
<be>

## Test
* **Run **`test.py`** to perform network inference and evaluation. Example for test [model_name] on [dataset_name] datasets:**
  ```
  python test.py --model_names FM-Net --dataset_names IRSTD-1K --save_img True
  ```
* **Predicted pictures will be saved to **`./results/`**, Pd and Fa will be saved to **`./test_ROC/`**:**
  ```
  ├──./results/
  │    ├── [dataset_name]
  │    │    ├── [model_name]
  |    |    |    |—— xxxxx.png
  |    |    |    |—— xxxxx.png
  ```
<be>

## Abstract
Infrared small target detection (IRSTD) aims to locate and separate targets from complex background. The challenges in IRSTD primarily come from extremely sparse target features and strong background clutter interference. However, existing methods typically perform discrimination directly on the features extracted by deep networks, neglecting the distinct characteristics of weak and small targets in the frequency domain, thereby limiting the improvement of detection capability. In this paper, we propose a frequency-aware masked-attention network (FM-Net) that leverages multi-scale frequency clues to assist in representing global context and suppressing noise interference. Specifically, we design the wavelet residual block (WRB) to extract multi-scale spatial and frequency features, which introduces a wavelet pyramid as the intermediate layer of the residual block. Then, to perceive global information on the long-range skip connections, a frequency-modulation masked-attention module (FMM) is used to interact multi-layer features from encoder. FMM contains two crucial elements: (a) a mask attention (MA) mechanism for injecting broad contextual feature efficiently to promote full-level semantic correlation and focus on salient regions, and (b) a channel-wise frequency modulation module (CFM) for enhancing the most informative frequency components and suppressing useless ones. Extensive experiments on three benchmark datasets (e.g. SIRST, NUDT-SIRST, IRSTD-1k) demonstrate that FM-Net achieves superior detection performance.

## Model
![Image text](https://github.com/yongxianLiu/FM-Net/blob/main/Fig/network.png)

## Results

#### Quantitative Results
| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|
| IRSTD-1K      | 68.13  | 95.62 | 8.085 |
| SIRST         | 76.75  | 96.96 | 15.710|
| NUDT-SIRST    | 94.76  | 99.15 | 1.838 |
| [[Weights]](https://pan.baidu.com/s/1QPzmKhNAMcwGVLiRvJvqjw?pwd=jqfb)|

#### Qualitative Results
![Image text](https://github.com/yongxianLiu/FM-Net/blob/main/Fig/vision2d.png)


## Citiation
```
@article{liu2025fm,
  title={FM-Net: Frequency-Aware Masked-Attention Network for Infrared Small Target Detection},
  author={Liu, Yongxian and Lin, Zaiping and Li, Boyang and Liu, Ting and An, Wei},
  journal={Remote Sensing},
  year={2025},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
<br>


## Acknowledgement
**Thanks for [SCTransNet](https://github.com/xdFai/SCTransNet), [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD), [WTAPNet](https://github.com/MinjieWan/WTAPNet), [FSNet](https://github.com/c-yn/FSNet).**
<br><br>

## Contact
**Welcome to raise issues or email to yongxian23@nudt.edu.cn for any question.**
