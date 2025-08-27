import xml.etree.ElementTree as ET


class ObjectItem:
    def __init__(self):
        self.points = []
        self.possible_result_name = ""
        self.score = None


class Annotation:
    def __init__(self):
        self.filename = ""
        self.objects = []


def create_voc_base():
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'VOC2007'

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'The VOC2007 Database'
    annotation_tag = ET.SubElement(source, 'annotation')
    annotation_tag.text = 'PASCAL VOC2007'
    image = ET.SubElement(source, 'image')
    image.text = 'flickr'
    flickrid = ET.SubElement(source, 'flickrid')
    flickrid.text = '341012865'

    owner = ET.SubElement(annotation, 'owner')
    flickrid_o = ET.SubElement(owner, 'flickrid')
    flickrid_o.text = 'Fried Camels'
    name_o = ET.SubElement(owner, 'name')
    name_o.text = 'Jinky the Fruit Bat'

    return annotation


def create_object_element(name, pose, truncated, difficult, bndbox, score):
    object_el = ET.Element('object')

    name_el = ET.SubElement(object_el, 'name')
    name_el.text = name

    pose_el = ET.SubElement(object_el, 'pose')
    pose_el.text = pose

    truncated_el = ET.SubElement(object_el, 'truncated')
    truncated_el.text = str(truncated)

    difficult_el = ET.SubElement(object_el, 'difficult')
    difficult_el.text = str(difficult)

    score_el = ET.SubElement(object_el, 'score')
    score_el.text = str(score)

    bndbox_el = ET.SubElement(object_el, 'bndbox')
    xmin_el = ET.SubElement(bndbox_el, 'xmin')
    xmin_el.text = str(int(bndbox[0]))
    ymin_el = ET.SubElement(bndbox_el, 'ymin')
    ymin_el.text = str(int(bndbox[1]))
    xmax_el = ET.SubElement(bndbox_el, 'xmax')
    xmax_el.text = str(int(bndbox[2]))
    ymax_el = ET.SubElement(bndbox_el, 'ymax')
    ymax_el.text = str(int(bndbox[3]))

    return object_el


def add_objects_to_voc(out_xml_path, boxes, types, scores):
    root = create_voc_base()

    # 添加 filename
    file_name = out_xml_path.split('/')[-1].split('.')[0] + '.jpg'
    filename_el = ET.SubElement(root, 'filename')
    filename_el.text = file_name

    # 添加 object
    for box, cls, score in zip(boxes, types, scores):
        object_el = create_object_element(cls, 'Unspecified', 0, 0, box, score)
        root.append(object_el)

    # 保存
    tree = ET.ElementTree(root)
    tree.write(out_xml_path, encoding='utf-8', xml_declaration=True)


def parse_xml_for_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = Annotation()
    annotation.filename = root.find('filename').text
    for obj in root.findall('object'):
        obj_item = ObjectItem()
        bandbox = obj.find('bndbox')
        xmin = int(bandbox.find('xmin').text is None and 0 or bandbox.find('xmin').text)
        ymin = int(bandbox.find('ymin').text is None and 0 or bandbox.find('ymin').text)
        xmax = int(bandbox.find('xmax').text is None and 0 or bandbox.find('xmax').text)
        ymax = int(bandbox.find('ymax').text is None and 0 or bandbox.find('ymax').text)
        obj_item.points.append(xmin)
        obj_item.points.append(ymin)
        obj_item.points.append(xmax)
        obj_item.points.append(ymax)
        score = obj.find('score')
        obj_item.score = float(score.text) if score is not None else ''
        name = obj.find('name')
        obj_item.possible_result_name = name.text
        annotation.objects.append(obj_item)
    return annotation


def add_objects_to_xml(input_xml_path, out_xml_path, boxes, types, scores, iou_thr):
    # 解析XML文件
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    voc_label = parse_xml_for_voc(input_xml_path)
    bboxs = [obj.points for obj in voc_label.objects]
    for box, cls, score in zip(boxes, types, scores):

        max_iou = calculate_max_iou(box, bboxs)
        if max_iou < iou_thr:
            # 创建object元素
            object_el = create_object_element(cls, 'Unspecified', 0, 0, box, score)
            # 将object元素添加到XML树中
            root.append(object_el)

    # 将修改后的XML保存到文件
    tree.write(out_xml_path)


def calculate_iou(box1, box2):
    # 计算两个边界框的交集部分
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 如果没有交集则返回0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou


def calculate_max_iou(bbox, bboxs):
    max_iou = 0
    for b in bboxs:
        iou = calculate_iou(bbox, b)
        if iou > max_iou:
            max_iou = iou
    return max_iou
