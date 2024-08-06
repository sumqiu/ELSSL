import cv2
import numpy as np

def load_image(path):
    """Load image"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img

def dice_coefficient(img1, img2):
    """Calculate the Dice coefficient"""
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    score = 2.0 * intersection.sum() / (img1.sum() + img2.sum())
    return score

def iou_coefficient(img1, img2):
    """Calculate the Intersection over Union (IoU) coefficient"""
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    score = intersection.sum() / float(union.sum())
    return score


def calculate_dice(path1, file_name1 , path2, file_name2):
    """Load an image from a given path and calculate the Dice coefficient"""
    img1 = load_image(path1 + '/' + file_name1)
    img2 = load_image(path2 + '/' + file_name2)
    return dice_coefficient(img1, img2)

def calculate_iou(path1, file_name1 , path2, file_name2):
    """Load an image from a specified path and calculate the Intersection over Union (IoU) coefficient"""
    img1 = load_image(path1 + '/' + file_name1)
    img2 = load_image(path2 + '/' + file_name2)
    return iou_coefficient(img1, img2)
