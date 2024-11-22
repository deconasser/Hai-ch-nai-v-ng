import os
import json
import cv2
import argparse

def yolo_to_coco(yolo_file, img_file, img_id, annotation_id, class_map):
    height, width, _ = cv2.imread(img_file).shape
    annotations = []
    
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width = bbox_width * width
            bbox_height = bbox_height * height

            annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": int(class_id),
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1
            
    return annotations, annotation_id

def convert_yolo_to_coco(yolo_dir, img_dir, class_map, output_file):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": class_name} for i, class_name in class_map.items()]
    }
    
    img_id = 1
    annotation_id = 1
    
    for img_file in os.listdir(img_dir):
        if img_file.endswith('.jpg'):
            yolo_file = os.path.join(yolo_dir, img_file.replace('.jpg', '.txt'))
            if os.path.exists(yolo_file):
                img_path = os.path.join(img_dir, img_file)
                height, width, _ = cv2.imread(img_path).shape
                
                coco_data["images"].append({
                    "id": img_id,
                    "file_name": img_file,
                    "height": height,
                    "width": width
                })
                
                annotations, annotation_id = yolo_to_coco(yolo_file, img_path, img_id, annotation_id, class_map)
                coco_data["annotations"].extend(annotations)
                
                img_id += 1
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO annotations saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format")
    parser.add_argument("--yolo_dir", required=True, help="Path to the YOLO annotations directory")
    parser.add_argument("--img_dir", required=True, help="Path to the image directory")
    parser.add_argument("--output_file", required=True, help="Path to the output COCO file (JSON)")
    
    args = parser.parse_args()

    class_map = {0: "motorcycle", 1: "car", 2: "touring_car", 3: "container"}
    
    # Run conversion
    convert_yolo_to_coco(args.yolo_dir, args.img_dir, class_map, args.output_file)
