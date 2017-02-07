import numpy as np
import matplotlib.pyplot as plt
# change this to your caffe root dir
caffe_root = '/home/jfg/Documents/work/caffe'
import os
import sys
sys.path.insert(0, caffe_root + '/python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found
    return labelnames


def ssd_detection(detect_image, map_label, is_show=True):
    model_def = 'models/VGGNet/LISA/SSD_414x125/deploy.prototxt'
    model_weights = 'models/VGGNet/LISA/SSD_414x125/VGG_LISA_SSD_414x125_iter_120000.caffemodel'
    if not os.path.exists('./predict_result'):
        os.mkdir('./predict_result')
    if len(detect_image.split('/')) != 0:
        image_save_path = './predict_result/' + detect_image.split('/')[-1]
    else:
        image_save_path = './predict_result/' + detect_image

    net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    image_resize_width = 414
    image_resize_height = 125
    net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
    image = caffe.io.load_image(detect_image)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_x1 = detections[0, 0, :, 3]
    det_y1 = detections[0, 0, :, 4]
    det_x2 = detections[0, 0, :, 5]
    det_y2 = detections[0, 0, :, 6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(map_label, top_label_indices)
    top_x1 = det_x1[top_indices]
    top_y1 = det_y1[top_indices]
    top_x2 = det_x2[top_indices]
    top_y2 = det_y2[top_indices]

    image_mat = cv2.imread(detect_image, cv2.IMREAD_COLOR)
    image_h = image_mat.shape[0]
    image_w = image_mat.shape[1]

    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_yellow = (0, 255, 255)
    color_text = (0, 0, 0)

    for i in range(0, top_conf.shape[0]):
        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        text_thickness = 1
        box_thickness = 1

        text_origin = (int(round(top_x1[i] * image_w)), int(round(top_y1[i] * image_h)))
        x1 = int(round(top_x1[i] * image_w))
        y1 = int(round(top_y1[i] * image_h))
        x2 = int(round(top_x2[i] * image_w))
        y2 = int(round(top_y2[i] * image_h))
        if 'Go' in top_labels[i]:
            cv2.rectangle(image_mat, (x1, y1), (x2, y2), color_green, box_thickness)
            text_size, base = cv2.getTextSize(top_labels[i], font_face, font_scale, text_thickness)
            cv2.rectangle(image_mat, text_origin, (text_origin[0]+text_size[0], text_origin[1]-text_size[1]),
                          color_green, cv2.FILLED)
            cv2.putText(image_mat, top_labels[i], text_origin,
                        font_face, font_scale, color_text, text_thickness, 1)
        elif 'Stop' in top_labels[i]:
            cv2.rectangle(image_mat, (x1, y1), (x2, y2), color_red, box_thickness)
            text_size, base = cv2.getTextSize(top_labels[i], font_face, font_scale, text_thickness)
            cv2.rectangle(image_mat, text_origin, (text_origin[0]+text_size[0], text_origin[1]-text_size[1]),
                          color_red, cv2.FILLED)
            cv2.putText(image_mat, top_labels[i], text_origin,
                        font_face, font_scale, color_text, text_thickness, 1)
        elif 'Warning' in top_labels[i]:
            cv2.rectangle(image_mat, (x1, y1), (x2, y2), color_yellow, box_thickness)
            text_size, base = cv2.getTextSize(top_labels[i], font_face, font_scale, text_thickness)
            cv2.rectangle(image_mat, text_origin, (text_origin[0]+text_size[0], text_origin[1]-text_size[1]),
                          color_yellow, cv2.FILLED)
            cv2.putText(image_mat, top_labels[i], text_origin,
                        font_face, font_scale, color_text, text_thickness, 1)
    cv2.imwrite(image_save_path, image_mat)
    if is_show:
        cv2.namedWindow('predict', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image_mat)

        cv2.waitKey(200)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    print('Check Caffe OK!')

    # load label map file
    map_file_name = 'data/labelmap_lisa.prototxt'
    map_file = open(map_file_name, 'r')
    label_map = caffe_pb2.LabelMap()
    a = text_format.Merge(str(map_file.read()), label_map)

    # test images
    ssd_detection('tl2.jpg', label_map, is_show=True)


