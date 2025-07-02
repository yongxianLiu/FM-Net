# Frequency-Aware Masked-Attention for Infrared Small Target Detection

Pytorch implementation of our "FM-Net: Frequency-Aware Masked-Attention for Infrared Small Target Detection". [[Paper]]()


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

## Model
![Image text](https://github.com/yongxianLiu/FM-Net/blob/main/Fig/network.png)

## Results
| Model         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| SIRST    | 77.50  |  81.08 | 87.32 |
| NUDT-SIRST    | 94.09  |  94.38 | 96.95 |
| IRSTD-1K      | 68.03  |  68.15 | 80.96 |
| [[Weights]](https://pan.baidu.com/s/1QPzmKhNAMcwGVLiRvJvqjw?pwd=jqfb)|


## Citiation
```
@article
```
<br>


## Acknowledgement
**Thanks for [SCTransNet](https://github.com/xdFai/SCTransNet), [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD), [WTAPNet](https://github.com/MinjieWan/WTAPNet).**
<br><br>

## Contact
**Welcome to raise issues or email to yongxian23@nudt.edu.cn for any question.**
