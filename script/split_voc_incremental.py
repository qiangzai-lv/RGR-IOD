import os
import xml.etree.ElementTree as ET
import argparse


ALL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

keep_list_count = {}
remove_list_count = {}


def write_list_to_file(lst, root):
    """写入列表到 txt 文件"""
    with open(root, 'w') as file:
        for item in lst:
            file.write(f"{item}\n")


def filter_objects_in_xml(xml_file_path, output_xml_file_path, keep_list):
    """
    解析 XML 文件，只保留 keep_list 中的类别
    """
    global keep_list_count, remove_list_count

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    objects_to_remove = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        if name not in keep_list:
            objects_to_remove.append(obj)
            remove_list_count[name] = remove_list_count.get(name, 0) + 1
        else:
            keep_list_count[name] = keep_list_count.get(name, 0) + 1

    for obj in objects_to_remove:
        root.remove(obj)

    if root.findall('object'):
        tree.write(output_xml_file_path)
        return True
    return False


def process_directory(input_txt, output_dir, keep_list, train_val_path, annotations_dir):
    """处理一个数据划分（trainval/test）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainval_list = []
    with open(input_txt, 'r', encoding='utf-8') as file:
        for line in file:
            filename = line.strip()
            input_xml_path = os.path.join(annotations_dir, filename + '.xml')
            output_xml_path = os.path.join(output_dir, filename + '.xml')

            if filter_objects_in_xml(input_xml_path, output_xml_path, keep_list):
                trainval_list.append(filename.rjust(6, '0'))

    write_list_to_file(trainval_list, train_val_path)


def main(args):
    global keep_list_count, remove_list_count

    base = int(args.pattern.split('+')[0])
    class_increment = int(args.pattern.split('+')[1])
    num_tasks = int((20 - base) / class_increment) + 1

    for i in range(num_tasks):
        keep_list_count = {}
        remove_list_count = {}

        if i == 0:
            keep_list = ALL_CLASSES[:base]

            process_directory(
                args.trainval_txt,
                args.output_dir.format(args.pattern, i, 'trainval'),
                keep_list,
                train_val_path=f'{args.output_dir.format(args.pattern, i, "trainval")}/../task{i}_trainval.txt',
                annotations_dir=args.annotations_dir
            )
            process_directory(
                args.test_txt,
                args.output_dir.format(args.pattern, i, 'test'),
                keep_list,
                train_val_path=f'{args.output_dir.format(args.pattern, i, "test")}/../task{i}_test.txt',
                annotations_dir=args.annotations_dir
            )

        else:
            keep_list_trainval = ALL_CLASSES[base + (i - 1) * class_increment: base + i * class_increment]
            keep_list_test = ALL_CLASSES[: base + i * class_increment]

            process_directory(
                args.trainval_txt,
                args.output_dir.format(args.pattern, i, 'trainval'),
                keep_list_trainval,
                train_val_path=f'{args.output_dir.format(args.pattern, i, "trainval")}/../task{i}_trainval.txt',
                annotations_dir=args.annotations_dir
            )
            process_directory(
                args.test_txt,
                args.output_dir.format(args.pattern, i, 'test'),
                keep_list_test,
                train_val_path=f'{args.output_dir.format(args.pattern, i, "test")}/../task{i}_test.txt',
                annotations_dir=args.annotations_dir
            )

        print(f"task{i}")
        print("keep list count:", keep_list_count)
        print("remove list count:", remove_list_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VOC 增量数据划分脚本")
    parser.add_argument('--pattern', type=str, default='10+10',
                        help="类别划分模式，例如 '10+10' 或 '15+1+1+1+1+1'")
    parser.add_argument('--annotations_dir', type=str,
                        default='data/VOCdevkit/VOC2007/Annotations',
                        help="VOC Annotations 文件夹路径")
    parser.add_argument('--trainval_txt', type=str,
                        default='data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                        help="trainval.txt 路径")
    parser.add_argument('--test_txt', type=str,
                        default='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
                        help="test.txt 路径")
    parser.add_argument('--output_dir', type=str,
                        default='data/VOCdevkit/VOC20007_split/{}/task{}_{}',
                        help="输出目录模板，支持格式化，例如 '.../{}/task{}_{}'")

    args = parser.parse_args()
    main(args)