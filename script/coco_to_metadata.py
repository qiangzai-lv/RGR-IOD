import json
from collections import defaultdict
from tqdm import tqdm


def coco_to_metadata(coco_json_path, metadata_jsonl_path):
    # 读取 COCO 标注文件
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 构建 id->name 的类别字典
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # 构建 image_id -> {file_name, categories}
    image_id_to_info = defaultdict(lambda: {"file_name": None, "categories": []})

    for img in coco["images"]:
        image_id_to_info[img["id"]]["file_name"] = img["file_name"]

    for ann in tqdm(coco["annotations"], desc="Processing annotations"):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name[cat_id]
        image_id_to_info[img_id]["categories"].append(cat_name)

    # 生成 metadata.jsonl
    with open(metadata_jsonl_path, "w", encoding="utf-8") as f:
        for img_id, info in image_id_to_info.items():
            if not info["categories"]:
                continue
            categories = sorted(set(info["categories"]))
            objects_str = ", ".join(categories)
            metadata = {
                "file_name": info["file_name"],
                "text": f"A realistic clear detailed photo of {objects_str}"
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print(f"metadata.jsonl saved at {metadata_jsonl_path}, total {len(image_id_to_info)} images")


if __name__ == "__main__":
    coco_json = "data/coco/annotations/instances_train2017.json"
    metadata_jsonl = "data/coco/train2017/metadata.jsonl"
    coco_to_metadata(coco_json, metadata_jsonl)