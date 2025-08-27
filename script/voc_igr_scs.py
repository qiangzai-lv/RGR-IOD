import os
import shutil
import random
import warnings
import argparse
from tqdm import tqdm

from add_object_voc import add_objects_to_voc
from mmdet.apis import inference_detector, init_detector

warnings.filterwarnings('ignore')

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


class Output:
    def __init__(self, xyxy: list, scores: list, cls: list, path):
        self.xyxy = xyxy
        self.scores = scores
        self.cls = cls
        self.path = path


class MmdetModel:
    def __init__(self, cfg_path, pt_path, class_names, skip_scores=0.5) -> None:
        self.cfg_path = cfg_path
        self.pt_path = pt_path
        self.skip_scores = skip_scores
        self.class_names = class_names
        self.model = init_detector(self.cfg_path, self.pt_path)

    def predict(self, img_path):
        result = inference_detector(self.model, img_path)
        labels = result.pred_instances.labels
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores
        ins = scores > self.skip_scores
        bboxes = bboxes[ins, :]
        labels = labels[ins]
        scores = scores[ins]
        return Output(
            bboxes.tolist(),
            scores.tolist(),
            [self.class_names[cls_idx] for cls_idx in labels],
            img_path
        )


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union


def calculate_max_iou(bbox, bboxs):
    return max([calculate_iou(bbox, b) for b in bboxs], default=0.0)


def select_and_copy_files(source_dir, target_dir, prefix, count=200):
    files = [f for f in os.listdir(source_dir) if f.startswith(prefix)]
    if len(files) < count:
        count = len(files)
    selected = random.sample(files, count)

    os.makedirs(target_dir, exist_ok=True)
    copied = []
    for file in selected:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def process_images(input_dir, out_img_dir, out_ann_dir,
                   gen_model_cfg, gen_model_pt, filter_model_cfg, filter_model_pt,
                   class_names, target_classes,
                   filter=True, filter_iou=0.5, filter_score=0.6, skip_scores=0.5):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    gen_model = MmdetModel(gen_model_cfg, gen_model_pt, class_names, skip_scores=skip_scores)
    filter_model = MmdetModel(filter_model_cfg, filter_model_pt, class_names, skip_scores=skip_scores)

    for fname in tqdm(os.listdir(input_dir)):
        cls_name = fname.split('_')[0]
        if cls_name not in target_classes:
            continue

        img_path = os.path.join(input_dir, fname)
        out_ann_path = os.path.join(out_ann_dir, fname.replace('.jpg', '.xml'))

        gen_res = gen_model.predict(img_path)
        filter_res = filter_model.predict(img_path)

        keep_flag = any(c == cls_name and s > 0.95 for c, s in zip(gen_res.cls, gen_res.scores))

        if keep_flag and filter:
            keep_flag = False
            for fbox, fcls, fscore in zip(filter_res.xyxy, filter_res.cls, filter_res.scores):
                iou = calculate_max_iou(fbox, gen_res.xyxy)
                if iou > filter_iou and fscore > filter_score:
                    keep_flag = True

        if keep_flag:
            shutil.copy2(img_path, out_img_dir)
            add_objects_to_voc(out_ann_path, gen_res.xyxy, gen_res.cls, gen_res.scores)


def parse_args():
    parser = argparse.ArgumentParser(description="Add predicted objects to VOC XML annotations with IoU filtering")
    parser.add_argument("--task_and_stage", type=str, default='voc_10_task0')
    parser.add_argument("--input_images", type=str, default='data/VOCdevkit/VOC2007/JPEGImages',
                        help="Path to input images")
    parser.add_argument("--skip_scores", type=float, default=0.5, help="Score threshold for predictions")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for filtering predictions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    task = args.task_and_stage.split('_')[0]
    stage = args.task_and_stage.split('_')[1]
    input_labels = 'data/VOCdevkit/VOC2007_split/' + task + '/' + stage + '_trainval'
    work_dir = 'faster-rcnn_r50_fpn_1x_voc_' + args.task_and_stage
    if stage.replace('task', '') != '0':
        work_dir = work_dir + '_rgr'

    gen_cfg = 'work_dirs/' + work_dir + '/' + work_dir + '.py'
    gen_pt = 'work_dirs/' + work_dir + '/' + 'epoch_12.pth'

    filter_cfg = gen_cfg  # 默认相同
    filter_pt = gen_pt

    output_labels_path = input_labels + '_pseudo'
    output_images_path = args.input_images + '_pseudo'

    process_images(args.input_images, output_images_path, output_labels_path,
                   gen_cfg, gen_pt, filter_cfg, filter_pt,
                   class_names=VOC_CLASSES,
                   target_classes=VOC_CLASSES,
                   filter=True, filter_iou=args.iou_thr, filter_score=args.skip_scores,
                   skip_scores=args.skip_scores)
