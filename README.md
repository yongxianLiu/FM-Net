# Frequency-Aware Masked-Attention for Infrared Small Target Detection

Pytorch implementation of our FM-Net


## Requirements
- Python 3.12
- pytorch 1.10.1, torchvision 0.11.2, torchaudio 0.10.1, Cuda 11.1 or higher
- ml_collections==1.1.0
- pytorch-wavelets==1.3.0
- torch-dct==0.1.6
- openmim==0.3.9, mmcv==2.2.0
- timm==1.0.15
<br><br>

## Datasets
* SIRST &nbsp; [[download dir]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* IRSTD-1K &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
* NUDT-SIRST &nbsp; [[download dir]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9864119)

## Train
```bash
python train.py
```
<br>

## Test
```bash
python test.py
```
<br>

## Citiation
```
@article
```
<br>


## Acknowledgement
**Thanks for [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection) and [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD).**
<br><br>

## Contact
**Welcome to raise issues or email to yongxian23@nudt.edu.cn for any question.**
