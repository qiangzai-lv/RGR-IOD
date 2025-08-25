import torch

from diffusers import DiffusionPipeline
import os

base_cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
incr_cls = ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

pipeline = DiffusionPipeline.from_pretrained("/home/Newdisk1/lvxq/RGR-IOD/pretrain/stable-diffusion-v1-5",
                                             torch_dtype=torch.float16)
# 加载 LoRA 权重
pipeline.load_lora_weights("/home/Newdisk1/lvxq/RGR-IOD/sd_coco_lora/checkpoint-35000")

# 设置 LoRA 权重的强度（默认 1.0，越大效果越明显）
pipeline.fuse_lora(lora_scale=1.0)

pipeline.to("cuda")

image_save_dir = 'data/COCO_sd——v3'
gan_num = 1000


def cls_gen(cls: []):
    for cls_name in cls:
        prompt = f"A realistic clear detailed photo of {cls_name}"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        for i in range(gan_num):
            save_img = os.path.join(image_save_dir, cls_name + '_' + str(i) + '.jpg')
            if os.path.exists(save_img):
                continue
            image = pipeline(prompt).images[0]
            image.save(save_img)


gen_1 = ['aeroplane', 'bicycle']
gen_2 = ['bird', 'boat']
gen_3 = ['bottle', 'bus']
gen_4 = ['car', 'cat']
gen_5 = ['chair', 'cow']
gen_6 = ['diningtable', 'dog']
gen_7 = ['horse', 'motorbike']
gen_8 = ['person', 'pottedplant']
gen_9 = ['sheep', 'sofa']
gen_10 = ['aeroplane', 'bicycle']

gen_10_1 = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
gen_10_2 = ['bus', 'car', 'cat', 'chair', 'cow']
gen_10_3 = ['diningtable', 'dog', 'horse', 'motorbike', 'person']
gen_10_4 = ['pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

if __name__ == '__main__':
    cls_gen(gen_1)
