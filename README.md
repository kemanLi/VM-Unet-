# VM-UNet
This is the official code repository for "VM-UNet: Vision Mamba UNet for Medical
Image Segmentation". {[Arxiv Paper](https://arxiv.org/abs/2402.02491)}

## Abstract
In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the foundation block to capture extensive contextual information, and an asymmetrical encoder-decoder structure is constructed. We conduct comprehensive experiments on the ISIC17, ISIC18, and Synapse datasets, and the results indicate that VM-UNet performs competitively in medical image segmentation tasks. To our best knowledge, this is the first medical image segmentation model constructed based on the pure SSM-based model. We aim to establish a baseline and provide valuable insights for the future development of more efficient and effective SSM-based segmentation systems.

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)}

## 1. Prepare the dataset

### DRIVE / STARE patch training

The retinal patch pipeline first splits the original training images, then
resizes images and masks to 576 x 576. Source validation folders are preserved
as final test sets, so overlapping patches from one original image cannot leak
between training and validation.

```bash
python datasets/prepare_retinal_576.py
```

The generated layout is:

```text
data/retinal_576/<DRIVE-or-STARE>/
  train/{images,masks}
  val/{images,masks}
  test/{images,masks}
  manifest.json
```

Training builds partially overlapping 192 x 192 patch candidates on a fixed
9 x 9 grid with stride 48, then samples and augments 4,800 patches online per
epoch. The complete grid is retained, including peripheral patches; FOV masks
do not remove training samples.
One deterministic field-of-view (FOV) mask is generated from each fundus
image without using its vessel annotation. Retinal validation/test metrics are
reported both inside the FOV (the primary metrics used for checkpoint
selection) and over the whole resized image. The paired image/mask
augmentation includes horizontal and vertical flips,
90-degree rotations, mild brightness/contrast changes, and gamma adjustment.
Validation and testing use 192 x 192 sliding windows with stride 96 and average
overlapping probabilities before computing full-image metrics. Architecture
ablations use BCE + Dice (1:1), AdamW, random initialization, a fixed learning
rate of 1e-4, batch size 32, threshold 0.5, and 150 epochs. There is no early
stopping or learning-rate schedule. VMamba pre-training remains available only
as an explicitly selected comparison.

```bash
python train.py --dataset DRIVE --model vmunet \
  --initialization scratch --run-tag baseline
python train.py --dataset STARE --model vmunet \
  --initialization scratch --run-tag baseline
```

To run the two retinal baselines sequentially on one GPU:

```bash
nohup bash run_retinal_baselines.sh > baseline_chain.log 2>&1 &
```

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing). For the retinal configuration, store `vmamba_small_e238_ema.pth` at `./pre_trained_weights/pre_trained_weights/vmamba_small_e238_ema.pth`.



## 3. Train the VM-UNet
```bash
cd VM-UNet
python train.py  # Train and test VM-UNet on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test VM-UNet on the Synapse dataset.
```

For retinal ablation experiments, the original model and the high-resolution
variant share the same data, loss, optimizer, and evaluation pipeline:

```bash
# Original VM-UNet baseline
python train.py --dataset DRIVE --model vmunet \
  --initialization scratch --run-tag baseline

# VM-UNet with the combined F0/F1 high-resolution module
python train.py --dataset DRIVE --model vmunet_highres \
  --initialization scratch --run-tag highres
```

`models/vmunet/vmunet.py` and `models/vmunet/vmamba.py` remain the original
baseline implementation. The high-resolution model is implemented separately
in `vmunet_highres.py` and `vmamba_highres.py`.

Architecture ablations default to random initialization. To reproduce a
separate VMamba-pre-trained run, opt in explicitly:

```bash
python train.py --dataset DRIVE --model vmunet \
  --initialization vmamba --run-tag pretrained_baseline
```

**NOTE**: If you want to use the trained checkpoint for inference testing only and save the corresponding test images, you can follow these steps:  

- **In `config_setting`**:  
   - Set the parameter `only_test_and_save_figs` to `True`.  
   - Fill in the path of the trained checkpoint in `best_ckpt_path`.  
   - Specify the save path for test images in `img_save_path`.  

- **Execute the script**:  
   After setting the above parameters, you can run `train.py`.

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Trained VM-UNet Checkpoint

- You can also obtain our trained VM-UNet on ISIC17, ISIC18 and Synapse from [Baidu Netdisk](https://pan.baidu.com/s/1lygUOFo6fMF_wS_dskwpBQ?pwd=5z00) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing).

## 6. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.


