import warnings
import os
import argparse
from tqdm import tqdm

from add_object_voc import add_objects_to_xml
from mmdet.apis import inference_detector, init_detector

warnings.filterwarnings('ignore')


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
        # COCO类别映射
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
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


def main(input_images_path, input_labels_path, out_labels_path, cfg, pt, skip_scores, iou_thr):
    model = MmdetModel(cfg, pt, skip_scores=skip_scores)

    if not os.path.exists(out_labels_path):
        os.makedirs(out_labels_path)

    for i in tqdm(os.listdir(input_labels_path)):
        if not i.endswith('.xml'):
            continue
        image_name = i.replace('.xml', '')
        img_path = os.path.join(input_images_path, image_name + '.jpg')
        if not os.path.exists(img_path):
            continue
        result = model.predict(img_path)
        input_xml_path = os.path.join(input_labels_path, i)
        output_xml_path = os.path.join(out_labels_path, i)
        add_objects_to_xml(input_xml_path, output_xml_path,
                           result.xyxy, result.cls, result.scores, iou_thr)


def parse_args():
    parser = argparse.ArgumentParser(description="Add predicted objects to VOC XML annotations with IoU filtering")
    parser.add_argument("--task_and_stage", type=str, default=' ')
    parser.add_argument("--input_images", type=str, default='data/VOCdevkit/VOC2007/JPEGImages', help="Path to input images")
    parser.add_argument("--skip_scores", type=float, default=0.5, help="Score threshold for predictions")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for filtering predictions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    task = args.task_and_stage.split('_')[0]
    stage = args.task_and_stage.split('_')[1]
    stage_id = int(stage.replace('task', ''))
    input_labels = 'data/VOCdevkit/VOC2007_split/' + task + '/' + stage + '_trainval'
    work_dir = 'faster-rcnn_r50_fpn_1x_voc_' + task + '_task' + str(stage_id - 1)

    if stage_id > 1 :
        work_dir = work_dir + '_rgr'
    mmdet_cfg = 'work_dirs/' + work_dir + '/' + work_dir + '.py'
    mmdet_pt = 'work_dirs/' + work_dir + '/' + 'epoch_12.pth'
    output_labels_path = input_labels + '_pseudo'
    main(args.input_images, input_labels, output_labels_path, mmdet_cfg, mmdet_pt, args.skip_scores, args.iou_thr)
