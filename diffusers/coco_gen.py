import torch

from diffusers import DiffusionPipeline
import os

base_cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
incr_cls = ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

pipeline = DiffusionPipeline.from_pretrained("/home/Newdisk1/lvxueqiang/diifusers/pretrain/sd1.5",
                                             torch_dtype=torch.float16)
pipeline.to("cuda")

image_save_dir = '/home/Newdisk1/lvxueqiang/data/coco_sd_add'
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


gen_1 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter']
gen_2 = ['bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella']
gen_3 = ['handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket']

gen_4 = ['bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog']
gen_5 = ['pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', ]
gen_6 = ['microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']

gen_7 = ['microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']

gen_8 = ['knife']

if __name__ == '__main__':
    cls_gen(gen_5)
