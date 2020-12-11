import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
__all__ = ['cv_show', 'add_chinese_text', 'is_image_file', 'is_dcm_file', 'has_file_allowed_extension']
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
DICOM_EXTENSIONS = ['dcm']


def cv_show(name, image, resize=1):
    """
    show image and resize show window
    :param name: window name
    :param image: image
    :param resize: resize ratio
    """
    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def list_image(path, format):
    """
    list image with the format you want
    :param path: image path
    :param format: image format you want
    :return: image list
    """
    list_images = [x for x in os.listdir(os.path.join(path)) if str(format) in x]
    return list_images


def add_chinese_text(frame, addtxt, left, bottom, color, textSize=20):
    """
    add chinese txt to image
    :param frame: image wanna to add
    :param addtxt: chinese txt
    :param left: txt left location
    :param bottom: txt bottom location
    :param fill: txt color
    :param textSize: text size
    :return: frame with chinese txt
    """
    fontpath = "./font/simsun.ttc"  # 宋体字体文件
    font = ImageFont.truetype(fontpath, textSize)  # 加载字体, 字体大小
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((left + 6, bottom - 20), addtxt, font=font, fill=color)
    frame = np.array(img_pil)
    return frame


def has_file_allowed_extension(filename, extensions):
    """
    Checks if a file is an allowed extension.
    :param filename: path to a file
    :param extensions:  extensions to consider (lowercase)
    :return:  bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """
    Checks if a file is an allowed image extension.
    :param filename: path to a file
    :return: bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def is_dcm_file(filename):
    """
    Checks if a file is an allowed dicom extension.
    :param filename: filename (string): path to a file
    :return: bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, DICOM_EXTENSIONS)