import sys
sys.path.append('.')
sys.path.append('../demo')
sys.path.append('/home/swang/work_space/caffe-c11/python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np
deploy = '12net.prototxt'
# caffemodel = './models/solver_iter_500000.caffemodel'
caffemodel = 'det1.caffemodel'
# caffemodel = '../demo/12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)



def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/127.5
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
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
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou') # 0.7

    return rectangles

threshold = [0.6,0.6,0.7]
imgpath = "../demo/test3.jpg"
rectangles = detectFace(imgpath,threshold)
img = cv2.imread(imgpath)
draw = img.copy()
for rectangle in rectangles:
    # cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    # for i in range(5,15,2):
    # 	cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
cv2.imshow("test",draw)
cv2.waitKey()
cv2.imwrite('test.jpg',draw)


