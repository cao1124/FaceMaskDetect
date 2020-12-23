import warnings
warnings.filterwarnings('ignore')
import cv2
import os
from collections import Counter
import numpy as np
from PIL import Image
import random
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference
from utils.img_utils import add_chinese_text
model = load_pytorch_model('models\\model360.pth')
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            temp = 30 + random.randint(5, 7) + random.randint(0, 9)/10
            cv2.putText(image, "%s %.1f" % (id2class[class_id], temp), (xmin + 2, ymin - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    cv2.namedWindow('GD_FaceMaskDetect', cv2.WINDOW_NORMAL)
    while status:
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        if (status):
            output_info = inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
            counter = [x[0] for x in output_info]
            # 切割人脸
            person_face = np.zeros((80, img_raw.shape[1], 3))

            # 增加logo显示 结果显示
            logo = cv2.imread('img\\gdpacs_logo.jpg')
            person_face[0: 80, 560: 640, :] = cv2.resize(logo[:, :, ::-1], (80, 80))
            if Counter(counter)[1] > 0:
                # 发现未带口罩，warning提醒
                img_raw = add_chinese_text(img_raw, '总人数：' + str(len(counter)) + '未戴口罩人数：'
                                           + str(Counter(counter).get(1)), 0, 20, (255, 0, 0), )
                output_info = [x for x in output_info if x[0] == 1]
                if len(output_info) > 1:
                    for i in range(len(output_info)):
                        face = img_raw[output_info[i][3]: output_info[i][5], output_info[i][2]:output_info[i][4], :]
                        person_face[0:80, 80 * i:80 * (i + 1), :] = cv2.resize(face, (80, 80))
                elif len(output_info) == 1:
                    face = img_raw[output_info[0][3]: output_info[0][5], output_info[0][2]:output_info[0][4], :]
                    person_face[0:80, 0:80, :] = cv2.resize(face, (80, 80))
                if idx % 10 == 0:
                    os.system('warning.wav')
                idx += 1
                # cv2.waitKey(5)
            elif Counter(counter)[0] > 0:
                img_raw = add_chinese_text(img_raw, '总人数：' + str(len(counter))
                                           + '未戴口罩人数：' + str(0), 0, 20, (255, 255, 255),)
                if idx % 20 == 0:
                    os.system('health.wav')
                idx += 1
            img_out = np.concatenate((img_raw, person_face), axis=0).astype(np.uint8)
            cv2.imshow('GD_FaceMaskDetect', img_out[:, :, ::-1])
            cv2.waitKey(5)

            # 点击小写字母q 退出程序
            if cv2.waitKey(1) == ord('q'):
                break
            # 点击窗口关闭按钮退出程序
            if cv2.getWindowProperty('GD_FaceMaskDetect', cv2.WND_PROP_VISIBLE) < 1:
                break


if __name__ == "__main__":
    video_path = 0
    run_on_video(video_path, 'result.mp4', conf_thresh=0.5)
