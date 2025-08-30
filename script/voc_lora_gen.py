import torch
from diffusers import DiffusionPipeline
import os
import argparse
from tqdm import tqdm

# VOC 20 类别
voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def cls_gen(pipeline, cls_list, save_dir, gan_num):
    for cls_name in cls_list:
        prompt = f"A realistic clear detailed photo of {cls_name}"

        for i in tqdm(range(gan_num), desc=f"Generating {cls_name}", unit="img"):
            save_img = os.path.join(save_dir, f"{cls_name}_{i}.jpg")
            if os.path.exists(save_img):
                continue
            image = pipeline(prompt).images[0]
            image.save(save_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate VOC images with Stable Diffusion + LoRA")
    parser.add_argument('--save_dir', type=str, default='data/SD_VOC20Images',
                        help='Directory to save generated images')
    parser.add_argument('--lora_weight', type=str, default='checkpoint-35000',
                        help='Directory to save generated images')
    parser.add_argument('--num', type=int, default=10000,
                        help='Number of images per class')

    args = parser.parse_args()

    # 加载模型
    pipeline = DiffusionPipeline.from_pretrained(
        "pretrain/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipeline.load_lora_weights("sd_coco_lora/" + args.lora_weight)
    pipeline.fuse_lora(lora_scale=1.0)
    pipeline.to("cuda")

    # 生成
    cls_gen(pipeline, voc_classes, args.save_dir, args.num)