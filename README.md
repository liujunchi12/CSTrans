# [CSTrans: Cross-Subdomain Transformer for Unsupervised Domain Adaptation]


### Environment (Python 3.7.13)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==6.2 -c pytorch
pip install tqdm==4.64.0
pip install tensorboard==2.11.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.7.3
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
```

### Pretrained ViT
Download the following models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)
- ViT-B_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-B_16.npz;tab=live_object)

### Datasets:

- Download [data](https://drive.google.com/file/d/1rnU49vEEdtc3EYVo7QydWzxcSuYqZbUB/view?usp=sharing) and replace the current `data/`

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`

### Training:

All commands can be found in `script.txt`.
```

```
Our code is largely borrowed from [TVT](https://github.com/uta-smile/TVT)
