import os
import shutil
import random
import warnings
import argparse
from tqdm import tqdm
from add_object_voc import add_objects_to_voc
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config

warnings.filterwarnings('ignore')


def parse_classes_from_cfg(cfg_path):
    cfg = Config.fromfile(cfg_path)
    assert 'classes' in cfg, 'cfg must contain "classes" key'
    classes = cfg['classes']
    return list(classes)


def generate_imageset_file(image_dir, output_txt):
    """生成 VOC 格式的 ImageSets txt 文件"""
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w") as f:
        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith(".jpg"):
                img_id = os.path.splitext(fname)[0]  # 去掉扩展名
                f.write(img_id + "\n")


class Output:
    def __init__(self, xyxy: list, scores: list, cls: list, path):
        self.xyxy = xyxy
        self.scores = scores
        self.cls = cls
        self.path = path


class MmdetModel:
    def __init__(self, cfg_path, pt_path, skip_scores=0.5) -> None:
        self.cfg_path = cfg_path
        self.pt_path = pt_path
        self.skip_scores = skip_scores
        self.class_names = parse_classes_from_cfg(cfg_path)
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


def process_images(input_dir, out_img_dir, out_ann_dir, imageset_txt, gen_num_per_cls,
                   gen_model_cfg, gen_model_pt, filter_model_cfg, filter_model_pt, filter_iou, filter_score,
                   skip_scores=0.7):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    print("Starting image generative processing pipeline.")
    print(f"Input directory: {input_dir}")
    print(f"Output image directory: {out_img_dir}")
    print(f"Output annotation directory: {out_ann_dir}")
    print(f"Target images per class: {gen_num_per_cls}")

    gen_model = MmdetModel(gen_model_cfg, gen_model_pt, skip_scores=skip_scores)
    filter_model = MmdetModel(filter_model_cfg, filter_model_pt, skip_scores=skip_scores)
    target_classes = gen_model.class_names

    print(f"Target images class: {target_classes}")

    # 建立类别 -> 图片路径列表
    target_classes_gen_count_filter = {cls: 0 for cls in target_classes}
    target_classes_gen_count_no_filter = {cls: 0 for cls in target_classes}
    gen_num_per_cls = gen_num_per_cls/2

    for fname in tqdm(os.listdir(input_dir)):
        f_cls_name = fname.split('_')[0]
        if f_cls_name not in target_classes:
            # print(f"Skipping file {fname}: class {f_cls_name} not in target classes.")
            continue
        if target_classes_gen_count_filter[f_cls_name] >= gen_num_per_cls:
            # print(f"Skipping file {fname}: reached generation limit for class {f_cls_name}.")
            continue

        img_path = os.path.join(input_dir, fname)
        out_ann_path = os.path.join(out_ann_dir, fname.replace('.jpg', '.xml'))

        gen_res = gen_model.predict(img_path)
        filter_res = filter_model.predict(img_path)

        igr_flag = any(c == f_cls_name and s > 0.9 for c, s in zip(gen_res.cls, gen_res.scores))

        scs_flag = any(
            calculate_max_iou(fbox, gen_res.xyxy) > filter_iou and fscore > filter_score
            for fbox, fcls, fscore in zip(filter_res.xyxy, filter_res.cls, filter_res.scores)
        )

        ps_save_flag = False

        if igr_flag:
            if target_classes_gen_count_no_filter[f_cls_name] < gen_num_per_cls:
                ps_save_flag = True
                target_classes_gen_count_no_filter[f_cls_name] += 1
                print(
                    f"Accepted no filter {fname} (class: {f_cls_name}). "
                    f"Progress: {target_classes_gen_count_no_filter[f_cls_name]}/{gen_num_per_cls}"
                )
            elif scs_flag and target_classes_gen_count_filter[f_cls_name] < gen_num_per_cls:
                print(
                    f"Accepted filter {fname} (class: {f_cls_name}). "
                    f"Progress: {target_classes_gen_count_filter[f_cls_name]}/{gen_num_per_cls}"
                )
                target_classes_gen_count_filter[f_cls_name] += 1
                ps_save_flag = True

        if ps_save_flag:
            shutil.copy2(img_path, out_img_dir)
            add_objects_to_voc(out_ann_path, gen_res.xyxy, gen_res.cls, gen_res.scores)


    generate_imageset_file(out_img_dir, imageset_txt)
    print("Image processing completed successfully.")
    print(f"Final generation counts ps no filter: {target_classes_gen_count_no_filter}")
    print(f"Final generation counts ps filter: {target_classes_gen_count_filter}")


def parse_args():
    parser = argparse.ArgumentParser(description="Add predicted objects to VOC XML annotations with IoU filtering")
    parser.add_argument("--task_and_stage", type=str, default='voc_10_task0')
    parser.add_argument("--input_images", type=str, default='data/SD_VOC20Images',
                        help="Path to input images")
    parser.add_argument("--filter_score", type=float, default=0.4, help="Score threshold for predictions")
    parser.add_argument("--filter_iou", type=float, default=0.4, help="IoU threshold for filtering predictions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    task = args.task_and_stage.split('_')[0]
    stage = args.task_and_stage.split('_')[1]
    stage_id = int(stage.replace('task', ''))

    work_dir_pre = 'faster-rcnn_r50_fpn_1x_voc_' + args.task_and_stage
    if stage.replace('task', '') != '0':
        work_dir_pre = work_dir_pre + '_rgr'

    # replay pre task
    pre_cfg = 'work_dirs/' + work_dir_pre + '/' + work_dir_pre + '.py'
    pre_pt = 'work_dirs/' + work_dir_pre + '/' + 'epoch_12.pth'

    work_dir_curr = 'faster-rcnn_r50_fpn_1x_voc_' + task + '_task' + str(stage_id + 1) + '_curr'

    # replay curr task
    curr_cfg = 'work_dirs/' + work_dir_curr + '/' + work_dir_curr + '.py'
    curr_pt = 'work_dirs/' + work_dir_curr + '/' + 'epoch_12.pth'

    print('Replay pre task images')
    output_labels_path = 'data/VOCdevkit/VOC2007_split/' + task + '/' + 'task' + str(stage_id + 1) + '_replay_old_ann'
    output_images_path = 'data/VOCdevkit/VOC2007_split/' + task + '/' + 'task' + str(stage_id + 1) + '_replay_old_image'
    imageset_txt = 'data/VOCdevkit/VOC2007_split/' + task + '/' + 'task' + str(stage_id + 1) + '_replay_old.txt'

    process_images(args.input_images, output_images_path, output_labels_path, gen_num_per_cls=200,
                   gen_model_cfg=pre_cfg, gen_model_pt=pre_pt, filter_model_cfg=curr_cfg, filter_model_pt=curr_pt,
                   filter_iou=args.filter_iou, filter_score=args.filter_score, imageset_txt=imageset_txt)

    print('Replay cur task images')
    output_labels_path = 'data/VOCdevkit/VOC2007_split/' + task + '/' + 'task' + str(stage_id + 1) + '_replay_curr_ann'
    output_images_path = 'data/VOCdevkit/VOC2007_split/' + task + '/' + 'task' + str(
        stage_id + 1) + '_replay_curr_image'
    imageset_txt = 'data/VOCdevkit/VOC2007_split/' + task + '/' + 'task' + str(stage_id + 1) + '_replay_curr.txt'

    process_images(args.input_images, output_images_path, output_labels_path, gen_num_per_cls=50,
                   gen_model_cfg=curr_cfg, gen_model_pt=curr_pt, filter_model_cfg=pre_cfg, filter_model_pt=pre_pt,
                   filter_iou=args.filter_iou, filter_score=args.filter_score, imageset_txt=imageset_txt)
