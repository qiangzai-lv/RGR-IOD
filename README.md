## Revisiting Generative Replay for Class Incremental Object Detection

Official Pytorch implementation for "Revisiting Generative Replay for Class Incremental Object Detection", CVPR 2025 Poster.

![image-20250818142604338](./assets/frame_work.jpg)

[[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Revisiting_Generative_Replay_for_Class_Incremental_Object_Detection_CVPR_2025_paper.html)]

## 🚀 Contributions

- Instead of developing a costly generative model for complex scenarios and multi-class instances in CIOD, we propose using an existing SD model for image-level generative replay for all tasks, preserving knowledge and bridging the gap between generated and real images.
- We propose to employ a SCS method to sift through and pinpoint more difficult samples across both old and new tasks. This approach substantially reduces false alarms for the new task while effectively retaining the previously acquired knowledge.
- We perform extensive experiments on the PASCAL VOC and MS COCO datasets under various settings, and our proposed approach attains state-of-the-art results when compared to other current methods.

## Get Started

- This repo is based on [MMDetection 3.3](https://github.com/open-mmlab/mmdetection)  [SD1.5](https://github.com/huggingface/diffusers). Please follow the installation of MMDetection [GETTING_STARTED.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) and make sure you can run it successfully.
```bash
conda create -n rgr-iod python=3.11 -y
conda activate rgr-iod
# Please follow the official MMDetection Installation Guide to set up this environment (PyTorch, MMCV, MMEngine, etc.).
cd our project
pip install -v -e .
# Create a new diffusers environment because it does not support lower versions of pytorch
conda create -n diffusers python=3.11 -y
conda activate diffusers
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
cd diffusers
pip install -v -e .
```

## Dataset

- Unzip COCO dataset into ./data/coco/
- Unzip VOC dataset into ./data/VOCdevkit/
- Run `python script/coco_to_metadata.py` to build stable diffusion fine-tuning data 
- Run 划分 `python script/split_voc_incremental.py --pattern 10+10` to split the VOC dataset 
## Pretrain

We use the following pretrained models in our framework:

- [**Stable Diffusion v1-5**](https://huggingface.co/runwayml/stable-diffusion-v1-5)

Please download the weights from HuggingFace and put them under the pretrain/ 

### **Fine-tuning Stable Diffusion**

We fine-tune **Stable Diffusion v1.5** on detection data to adapt it to the style of object detection for generative replay.

```bash
conda activate diffusers
bash script/finetune_sd_coco_lora.sh
# Generate images at one time to reduce training costs
python script/voc_lora_gen.py 
```


## Train
```bash
# assume that you are under the root directory of this project,
# Two-step(10+10)
conda activate rgr-iod
bash ./tools/dist_train.sh configs/rgr_iod/faster-rcnn_r50_fpn_1x_voc_10+10_task0.py 4   # train base 10 cats
bash ./tools/dist_train.sh configs/rgr_iod/faster-rcnn_r50_fpn_1x_voc_10+10_task1_curr.py 4   # train curr 10 cats
python script/pseudo_label_voc.py --task_and_stage 10+10_task1 # pseudo_label
python script/voc_igr_scs.py --task_and_stage 10+10_task0 # generative replay
bash ./tools/dist_train.sh configs/rgr_iod/faster-rcnn_r50_fpn_1x_voc_10+10_task1_rgr.py 4   # train incr 10 cats
```

## Acknowledgement
Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).

## Citation
Please cite our paper if this repo helps your research:

```bibtex
@InProceedings{Zhang_2025_CVPR,
    author    = {Zhang, Shizhou and Lv, Xueqiang and Xing, Yinghui and Wu, Qirui and Xu, Di and Zhang, Yanning},
    title     = {Revisiting Generative Replay for Class Incremental Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20340-20349}
}
```









