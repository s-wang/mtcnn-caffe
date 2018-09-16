import sys
sys.path.append('../demo/')
sys.path.append('../prepare_data/')
sys.path.append('.')
sys.path.append('/home/swang/work_space/caffe-c11/python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np
import os
from utils import *
deploy = '../12net/12net.prototxt'
caffemodel = '../12net/models/solver_iter_100000.caffemodel'
# caffemodel = '../demo/det1.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)


def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()

def detectFace(img,threshold):
    caffe_img = (img.copy()-127.5) / 128
    origin_h,origin_w,ch = caffe_img.shape
    # calculate scales
    scales = tools.calculateScales(img)
    out = []
    # compute forward for each scale
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
        caffe.set_device(0)
        caffe.set_mode_gpu()
        out_ = net_12.forward()
        out.append(out_)

    # detect face based on forward results
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        reg      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,reg,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        if len(rectangle) == 0:
            continue
        keep_idx = py_nms(rectangle[:, :5], 0.5, 'Union')
        rectangle_keep = rectangle[keep_idx]

        rectangles.extend(rectangle_keep)

    # merge the detection from first stage
    rectangles = np.vstack(rectangles)
    keep_idx = py_nms(rectangles[:, 0:5], 0.7, 'Union')
    rectangle_keep = rectangles[keep_idx]
    boxes = rectangle_keep[:, :5]

    bbw = rectangle_keep[:, 2] - rectangle_keep[:, 0] + 1
    bbh = rectangle_keep[:, 3] - rectangle_keep[:, 1] + 1

    # refine the boxes based on regression results
    boxes_c = np.vstack([rectangle_keep[:, 0] + rectangle_keep[:, 5] * bbw,
                         rectangle_keep[:, 1] + rectangle_keep[:, 6] * bbh,
                         rectangle_keep[:, 2] + rectangle_keep[:, 7] * bbw,
                         rectangle_keep[:, 3] + rectangle_keep[:, 8] * bbh,
                         rectangle_keep[:, 4]])
    boxes_c = boxes_c.T

    # rectangles = tools.NMS(rectangles, 0.7, 'iou')
    return boxes_c

anno_file = '../prepare_data/wider_face_train.txt'
im_dir = "/home/swang/work_space/dataset/Wider_Face/WIDER_train/images/"

threshold = [0.6,0.6,0.7]
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print ("%d pics in total" % num)

image_idx = 0

num_tp = 0.
num_gt = 0.

for annotation in annotations:
    image_idx += 1
    if (image_idx - 1) % 100 !=0:
        continue

    annotation = annotation.strip().split(' ')
    bbox = list(map(float, annotation[1:]))
    gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    num_gt += len(gts)

    img_path = im_dir + annotation[0] + '.jpg'
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rectangles = detectFace(img,threshold)

    # change to square
    rectangles = convert_to_square(rectangles)
    rectangles[:, 0:4] = np.round(rectangles[:, 0:4])

    # draw = img.copy()
    # for rectangle in rectangles:
    #     cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)
    #
    # cv2.imshow("test", draw)
    # cv2.waitKey()

    view_bar(image_idx,num)

    for box in rectangles:
        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gts)

        if np.max(Iou) >= 0.65:
            num_tp += 1

recall = num_tp / num_gt
print("recall:", recall)
