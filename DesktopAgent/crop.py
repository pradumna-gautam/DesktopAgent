import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import clip
import torch

def crop_image(img, position):
    """
    Crop an image based on specified position coordinates.
    :param img: Input image.
    :param position: Coordinates for cropping.
    :return: Cropped image.
    """
    def distance(x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if position[i][0] > position[j][0]:
                position[i], position[j] = position[j], position[i]
    if position[0][1] > position[1][1]:
        position[0], position[1] = position[1], position[0]
    if position[2][1] > position[3][1]:
        position[2], position[3] = position[3], position[2]

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.float32)
    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.array([[0, 0], [img_width-1, 0], [0, img_height-1], [img_width-1, img_height-1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst

def calculate_size(box):
    """
    Calculate the size of a bounding box.
    :param box: Bounding box coordinates.
    :return: Size of the bounding box.
    """
    return (box[2] - box[0]) * (box[3] - box[1])

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    :param box1: First bounding box coordinates.
    :param box2: Second bounding box coordinates.
    :return: IoU value.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = calculate_size(box1)
    box2Area = calculate_size(box2)
    unionArea = box1Area + box2Area - interArea
    iou = interArea / unionArea
    
    return iou

def crop(image, box, i, text_data=None):
    """
    Crop an image based on a bounding box and save it.
    :param image: Path to the image.
    :param box: Bounding box coordinates.
    :param i: Index for saving the cropped image.
    :param text_data: Optional text data for annotation.
    """
    image = Image.open(image)
    
    if text_data:
        draw = ImageDraw.Draw(image)
        draw.rectangle(((text_data[0], text_data[1]), (text_data[2], text_data[3])), outline="red", width=5)
        # Optional: Uncomment to add text annotation
        # font_size = int((text_data[3] - text_data[1]) * 0.75)
        # font = ImageFont.truetype("arial.ttf", font_size)
        # draw.text((text_data[0]+5, text_data[1]+5), str(i), font=font, fill="red")

    cropped_image = image.crop(box)
    cropped_image.save(f"./temp/{i}.jpg")

def in_box(box, target):
    """
    Check if one box is completely inside another.
    :param box: Inner box coordinates.
    :param target: Outer box coordinates.
    :return: Boolean indicating if the box is inside the target.
    """
    return (box[0] > target[0]) and (box[1] > target[1]) and (box[2] < target[2]) and (box[3] < target[3])

def crop_for_clip(image, box, i, position):
    """
    Crop an image for CLIP model based on a specified position.
    :param image: Path to the image.
    :param box: Bounding box coordinates.
    :param i: Index for saving the cropped image.
    :param position: Position for cropping (left, right, top, bottom, etc.).
    :return: Boolean indicating if the box is within the specified position.
    """
    image = Image.open(image)
    w, h = image.size
    if position == "left":
        bound = [0, 0, w/2, h]
    elif position == "right":
        bound = [w/2, 0, w, h]
    elif position == "top":
        bound = [0, 0, w, h/2]
    elif position == "bottom":
        bound = [0, h/2, w, h]
    elif position == "top left":
        bound = [0, 0, w/2, h/2]
    elif position == "top right":
        bound = [w/2, 0, w, h/2]
    elif position == "bottom left":
        bound = [0, h/2, w/2, h]
    elif position == "bottom right":
        bound = [w/2, h/2, w, h]
    else:
        bound = [0, 0, w, h]
    
    if in_box(box, bound):
        cropped_image = image.crop(box)
        cropped_image.save(f"./temp/{i}.jpg")
        return True
    else:
        return False

def clip_for_icon(clip_model, clip_preprocess, images, prompt):
    """
    Use the CLIP model to find the best matching image for a prompt.
    :param clip_model: The CLIP model.
    :param clip_preprocess: Preprocessing function for the CLIP model.
    :param images: List of image file paths.
    :param prompt: Text prompt for matching.
    :return: Index of the best matching image.
    """
    image_features = []
    for image_file in images:
        image = clip_preprocess(Image.open(image_file)).unsqueeze(0).to(next(clip_model.parameters()).device)
        image_feature = clip_model.encode_image(image)
        image_features.append(image_feature)
    image_features = torch.cat(image_features)
    
    text = clip.tokenize([prompt]).to(next(clip_model.parameters()).device)
    text_features = clip_model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=0).squeeze(0)
    _, max_pos = torch.max(similarity, dim=0)
    pos = max_pos.item()
    
    return pos
