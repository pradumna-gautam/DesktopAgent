from DesktopAgent.crop import calculate_size, calculate_iou  # Assuming the desktop-specific module
from modelscope.pipelines import pipeline
from PIL import Image
import torch

def remove_boxes(boxes_filt, size, iou_threshold=0.5):
    boxes_to_remove = set()

    for i in range(len(boxes_filt)):
        if calculate_size(boxes_filt[i]) > 0.02*size[0]*size[1]:  # Adjust threshold for desktop
            boxes_to_remove.add(i)
        for j in range(len(boxes_filt)):
            if calculate_size(boxes_filt[j]) > 0.02*size[0]*size[1]:  # Adjust threshold for desktop
                boxes_to_remove.add(j)
            if i == j:
                continue
            if i in boxes_to_remove or j in boxes_to_remove:
                continue
            iou = calculate_iou(boxes_filt[i], boxes_filt[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    boxes_filt = [box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove]
    
    return boxes_filt


def det(input_image_path, caption, groundingdino_model, box_threshold=0.05, text_threshold=0.5):
    image = Image.open(input_image_path)
    size = image.size

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    
    inputs = {
        'IMAGE_PATH': input_image_path,
        'TEXT_PROMPT': caption,
        'BOX_THRESHOLD': box_threshold,
        'TEXT_THRESHOLD': text_threshold
    }

    result = groundingdino_model(inputs)
    boxes_filt = result['boxes']

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu().int().tolist()
    filtered_boxes = remove_boxes(boxes_filt, size)
    
    coordinate = []
    image_data = []
    for box in filtered_boxes:
        # Adjust padding for desktop
        image_data.append([max(0, box[0]-20), max(0, box[1]-20), min(box[2]+20, size[0]), min(box[3]+20, size[1])])
        coordinate.append([max(0, box[0]-30), max(0, box[1]-30), min(box[2]+30, size[0]), min(box[3]+30, size[1])])

    return image_data, coordinate
