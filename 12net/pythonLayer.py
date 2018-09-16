import sys
sys.path.append('/home/swang/work_space/caffe-c11/python')
import caffe
import cv2
import numpy as np
import random
import pickle as pickle
imdb_exit = True

random.seed(6)


def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()

################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        print("in setup")
        self.batch_size = 384 # 64
        net_side = 12
        self.batch_loader = BatchLoader(net_side)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 1)
        top[2].reshape(self.batch_size, 4)
        top[3].reshape(self.batch_size, 10)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, label, roi, pts = self.batch_loader.load_next_image()
            # im, label, roi = self.batch_loader.load_next_image()
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
            top[2].data[itt, ...] = roi
            top[3].data[itt, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,net_side):

        print("Start Reading Train Data into Memory...")
        if imdb_exit:
            fid = open('../prepare_data/12/12_all.imdb','rb')
            self.train_list = pickle.load(fid)
            fid.close()

        random.shuffle(self.train_list)
        self.cur = 0
        print("\n",str(len(self.train_list))," Train Data have been read into Memory...")

    def load_next_image(self, ):
        if self.cur == len(self.train_list):
            self.cur = 0
            random.shuffle(self.train_list)
        cur_data = self.train_list[self.cur]  # Get the image index
        im       = cur_data[0]
        label    = cur_data[1]
        roi = cur_data[2]
        pts = cur_data[3]
        # if label == 1 and random.choice([0,1]) == 1:
        #     im = cv2.flip(im, 1)
        if random.choice([0,1]) == 1:
            if label == 1:
                im = cv2.flip(im, 1)
                roi[0] = roi[0] * -1.0
                roi[2] = roi[2] * -1.0
            # if label == 0:
            #     im = cv2.flip(im, 1)
            # if label == -2:
            #     im = cv2.flip(im, 1)
            #     pts[0] = 1 - pts[0]
            #     pts[2] = 1 - pts[2]
            #     pts[4] = 1 - pts[4]
            #     pts[6] = 1 - pts[6]
            #     pts[8] = 1 - pts[8]
            #     t = pts[0]; pts[0] = pts[2]; pts[2] = t
            #     t = pts[1]; pts[1] = pts[3]; pts[3] = t
            #     t = pts[6]; pts[6] = pts[8]; pts[8] = t
            #     t = pts[7]; pts[7] = pts[9]; pts[9] = t
        self.cur += 1
        return im, label, roi, pts
        # return im, label, roi


################################################################################
#########################ROI Loss Layer By Python###############################
################################################################################
class regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")
        roi = bottom[1].data
        self.valid_index = np.where(roi[:,0] != -1)[0]
        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            top[0].data[...] = np.sum(self.diff**2) / bottom[0].num


    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i] or self.N==0:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num

################################################################################
#############################SendData Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        label = bottom[1].data
        self.valid_index = np.where(label >= 0)[0]
        self.count = len(self.valid_index)
        # count_1 = len(np.where(label == -1)[0])
        # count1 = len(np.where(label == 1)[0])
        # count0 = len(np.where(label == 0)[0])
        # print("count-1, count1, count0", count_1, count1, count0)
        # print("label", label[self.valid_index])
        # print("valid_index", self.valid_index)
        # print("cls valid num", self.count)
        # print("len(bottom[1].data", len(bottom[1].data))
        # top[0].reshape(len(bottom[1].data), 2,1,1)
        # top[1].reshape(len(bottom[1].data), 1)
        top[0].reshape(self.count, 2,1,1)
        top[1].reshape(self.count, 1)

    def forward(self,bottom,top):
        top[0].data[...][...]=0
        top[1].data[...][...]=0
        top[0].data[...] = bottom[0].data[self.valid_index]
        top[1].data[...] = bottom[1].data[self.valid_index]
        # top[0].data[0:self.count] = bottom[0].data[self.valid_index]
        # top[1].data[0:self.count] = bottom[1].data[self.valid_index]
        # top[0].data[...] = bottom[0].data[...]
        # top[1].data[...] = bottom[1].data[...]


    def backward(self,top,propagate_down,bottom):
        if propagate_down[0] and self.count!=0:
            bottom[0].diff[...]=0
            bottom[0].diff[self.valid_index]=top[0].diff[...]
        if propagate_down[1] and self.count!=0:
            bottom[1].diff[...]=0
            bottom[1].diff[self.valid_index]=top[1].diff[...]

class reg_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        # label = bottom[2].data
        roi = bottom[1].data
        # self.valid_index = np.where(label != 0)[0]
        self.valid_index = np.where(roi[:,0] != -1)[0]
        self.count = len(self.valid_index)
        # print("label", label)
        # print("valid_index", self.valid_index)
        # print(self.count)
        top[0].reshape(self.count, 4,1,1)
        top[1].reshape(self.count, 4)

    def forward(self,bottom,top):
        top[0].data[...][...]=0
        top[1].data[...][...]=0
        top[0].data[...] = bottom[0].data[self.valid_index]
        top[1].data[...] = bottom[1].data[self.valid_index]
        # top[0].data[...] = bottom[0].data[...]
        # top[1].data[...] = bottom[1].data[...]


    def backward(self,top,propagate_down,bottom):
        if propagate_down[0] and self.count!=0:
            bottom[0].diff[...]=0
            bottom[0].diff[self.valid_index]=top[0].diff[...]
        if propagate_down[1] and self.count!=0:
            bottom[1].diff[...]=0
            bottom[1].diff[self.valid_index]=top[1].diff[...]



class pts_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        # label = bottom[2].data
        pts = bottom[1].data
        # self.valid_index = np.where(label != 0)[0]
        self.valid_index = np.where(pts[:,0] != -1)[0]
        self.count = len(self.valid_index)
        # print("pts", pts)
        # print("valid_index", self.valid_index)
        # print(self.count)
        top[0].reshape(self.count, 10,1,1)
        top[1].reshape(self.count, 10)

    def forward(self,bottom,top):
        top[0].data[...][...]=0
        top[1].data[...][...]=0
        top[0].data[...] = bottom[0].data[self.valid_index]
        top[1].data[...] = bottom[1].data[self.valid_index]
        # top[0].data[...] = bottom[0].data[...]
        # top[1].data[...] = bottom[1].data[...]


    def backward(self,top,propagate_down,bottom):
        if propagate_down[0] and self.count!=0:
            bottom[0].diff[...]=0
            bottom[0].diff[self.valid_index]=top[0].diff[...]
        if propagate_down[1] and self.count!=0:
            bottom[1].diff[...]=0
            bottom[1].diff[self.valid_index]=top[1].diff[...]