from mmdet.apis import init_detector, inference_detector
import os
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion

# Đường dẫn đến file config và checkpoint của các model CO-DETR
model_configs = [
    ("./nextgen-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./nextgen/epoch_2.pth"),
    ("./nextgen-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./nextgen/epoch_6.pth"),
    ("./nextgen-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./nextgen/epoch_8.pth"),
    ("./nextgen-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./nextgen/epoch_10.pth"),
]

# Model ban đêm
night_model_config = ("./nextgen-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./nextgen/epoch_4(1).pth")

# Model ban ngày
day_model_config = ("./nextgen-cfg/co_dino_5scale_swin_large_16e_o365tococo.py", "./nextgen/epoch_12(1).pth")

# Danh sách các scale khác nhau cho từng model
scales = [(3.0, 3.0), (1.5, 1.5), (0.75, 0.75), (2.0, 2.0)]

input_folder = "./test_data/public test"
output_file_path = "./predict.txt"

# Kích thước ảnh thực tế
image_width = 1280
image_height = 720

# Khởi tạo mô hình
models = [init_detector(config, checkpoint, device='cuda:0') for config, checkpoint in model_configs]
night_model = init_detector(night_model_config[0], night_model_config[1], device='cuda:0')
day_model = init_detector(day_model_config[0], day_model_config[1], device='cuda:0')

# Kiểm tra nếu file tồn tại trước đó, xóa đi để tạo file mới
if os.path.exists(output_file_path):
    os.remove(output_file_path)

# Hàm chuyển đổi kết quả cho WBF, lọc các bounding box có confidence > 0.1
def format_result_for_wbf(result, scaled_width, scaled_height, score_threshold=0.015, flip=False):
    boxes, scores, labels = [], [], []
    for class_id, bboxes in enumerate(result):
        for bbox in bboxes:
            x1, y1, x2, y2, confidence_score = bbox
            if confidence_score > score_threshold:  # Lọc theo threshold
                if flip:  # Nếu là ảnh đã lật, đảo chiều bounding box
                    x1, x2 = 1 - x2 / scaled_width, 1 - x1 / scaled_width
                else:
                    x1, x2 = x1 / scaled_width, x2 / scaled_width
                y1, y2 = y1 / scaled_height, y2 / scaled_height
                boxes.append([x1, y1, x2, y2])
                scores.append(confidence_score)
                labels.append(class_id)
    return boxes, scores, labels

# Hàm thực hiện TTA và trả kết quả từ các augmentations
def tta_inference(model, image, scales, image_width, image_height):
    all_boxes, all_scores, all_labels = [], [], []

    for scale_w, scale_h in scales:
        scaled_width = int(image_width * scale_w)
        scaled_height = int(image_height * scale_h)
        scaled_image = cv2.resize(image, (scaled_width, scaled_height))

        result_original = inference_detector(model, scaled_image)
        boxes, scores, labels = format_result_for_wbf(result_original, scaled_width, scaled_height)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

        flipped_image = cv2.flip(scaled_image, 1)
        result_flipped = inference_detector(model, flipped_image)
        boxes, scores, labels = format_result_for_wbf(result_flipped, scaled_width, scaled_height, flip=True)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY))
        equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        result_equalized = inference_detector(model, equalized_image)
        boxes, scores, labels = format_result_for_wbf(result_equalized, scaled_width, scaled_height)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels

# Mở file để ghi kết quả
try:
    with open(output_file_path, 'w') as f:
        for image_file in os.listdir(input_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(input_folder, image_file)
                original_image = cv2.imread(image_path)

                # Phát hiện ảnh ban ngày/ban đêm dựa trên tên file
                is_image_night = '.rf.' in image_file

                # Danh sách model sẽ sử dụng
                active_models = models[:]  # Sao chép danh sách model cơ bản
                active_scales = scales[:]
                
                if is_image_night:
                    active_models.append(night_model)
                    active_scales.append((2.0, 2.0))  
                else:
                    active_models.append(day_model)
                    active_scales.append((2.0, 2.0))  

                # Danh sách để lưu kết quả inference từ các mô hình với TTA
                all_boxes, all_scores, all_labels = [], [], []

                for model, scale in zip(active_models, active_scales):
                    tta_boxes, tta_scores, tta_labels = tta_inference(
                        model, original_image, [scale], image_width, image_height
                    )
                    all_boxes.extend(tta_boxes)
                    all_scores.extend(tta_scores)
                    all_labels.extend(tta_labels)

                # Áp dụng WBF để kết hợp kết quả
                if all_boxes:
                    boxes, scores, labels = weighted_boxes_fusion(
                        all_boxes, all_scores, all_labels, iou_thr=0.7, skip_box_thr=0.015
                    )

                    for box, score, label in zip(boxes, scores, labels):
                        if score > 0.015:
                            x_center = (box[0] + box[2]) / 2.0
                            y_center = (box[1] + box[3]) / 2.0
                            width = box[2] - box[0]
                            height = box[3] - box[1]

                            line = f"{image_file} {int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                            f.write(line)

except Exception as e:
    print(f"Lỗi khi ghi vào file: {e}")

print(f"Kết quả đã được ghi vào file: {output_file_path}")
