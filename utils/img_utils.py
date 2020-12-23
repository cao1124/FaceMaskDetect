import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
