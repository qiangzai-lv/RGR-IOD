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
conda create -n rgr-iod python=3.9.21 -y
conda activate rgr-iod
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
pip install mmcv==2.0.1 --use-pep517 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
cd our project
pip install -v -e .
cd diffusers
pip install -v -e .
```

## Dataset

- Unzip COCO dataset into ./data/coco/
- Unzip VOC dataset into ./data/VOCdevkit/
- Run `./script/coco_to_metadata.py` to build stable diffusion fine-tuning data 
- Run 划分 `python split_voc_incremental.py --pattern 10+10` to split the VOC dataset 
## Checkpoints






## Train
```python
# assume that you are under the root directory of this project,

# Two-step(70+10)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/70+10/gdino_inc_70+10_0-69_scratch_coco.py 4   # train first 70 cats
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py 4 --amp # train last 10 cats incrementally

# Multi-step(40+10*4)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/40+40/gdino_inc_40+40_0-39_scratch_coco.py 4   
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/40+10_4/gdino_inc_40+10_4_40-49_gcd_scratch_coco.py 4 --amp
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/40+10_4/gdino_inc_40+10_4_50-59_gcd_scratch_coco.py 4 --amp
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/40+10_4/gdino_inc_40+10_4_60-69_gcd_scratch_coco.py 4 --amp
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/gdino_inc/40+10_4/gdino_inc_40+10_4_70-79_gcd_scratch_coco.py 4 --amp 
```

## Test
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_test.sh ./configs/gdino_inc/70+10/gdino_inc_70+10_70-79_gcd_scratch_coco.py ./work_dirs/gdino_inc_70+10_70-79_gcd_scratch_coco/epoch_12.pth 4 --cfg-options test_evaluator.classwise=True
```

## Acknowledgement
Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).
Thanks to the work [ERD](https://github.com/Hi-FT/ERD) and [CL-DETR](https://github.com/yaoyao-liu/CL-DETR).

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









