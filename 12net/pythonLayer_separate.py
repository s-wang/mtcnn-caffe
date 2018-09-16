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
        self.batch_size = 64 # 64
        net_side = 12
        self.batch_loader = BatchLoader(net_side)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 1)
        top[2].reshape(self.batch_size, 4)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # loss_task = random.randint(0,1)
        # loss_task = np.random.choice([0, 1], 1, p=[0.1, 0.9])
        # loss_task = 0
        for itt in range(self.batch_size):
            # loss_task = np.random.choice([0, 1], 1, p=[0.8, 0.2])
            loss_task = itt % 5
            # loss_task = 1
            im, label, roi= self.batch_loader.load_next_image(loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
            top[2].data[itt, ...] = roi

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,net_side):

        print("Start Reading Classification Data into Memory...")
        if imdb_exit:
            fid = open('../prepare_data/12/cls.imdb','rb')
            self.cls_list = pickle.load(fid)
            fid.close()

        random.shuffle(self.cls_list)
        self.cls_cur = 0
        print("\n",str(len(self.cls_list))," Train Data have been read into Memory...")


        print("Start Reading Regression Data into Memory...")
        if imdb_exit:
            fid = open('../prepare_data/12/roi_pos-1.imdb','rb')
            self.roi_list = pickle.load(fid)
            fid.close()
        random.shuffle(self.roi_list)
        self.roi_cur = 0
        print("\n",str(len(self.roi_list))," Regression Data have been read into Memory...")


    def load_next_image(self, loss_task):

        if loss_task != 0: #cls
            if self.cls_cur == len(self.cls_list):
                self.cls_cur = 0
                random.shuffle(self.cls_list)
            cur_data = self.cls_list[self.cls_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            roi      = [-1,-1,-1,-1]
            if random.choice([0,1]) == 1:
                if label == 0:
                    im = cv2.flip(im,random.choice([-1,0,1]))
                else:
                    im = cv2.flip(im, 1)
                    # im = cv2.flip(im, random.choice([-1, 0, 1]))
            self.cls_cur += 1
            return im, label, roi

        if loss_task == 0: #reg
            if self.roi_cur == len(self.roi_list):
                self.roi_cur = 0
                random.shuffle(self.roi_list)
            cur_data = self.roi_list[self.roi_cur]  # Get the image index
            im	     = cur_data[0]
            label    = -1
            roi      = cur_data[2]
            self.roi_cur += 1
            return im, label, roi


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
        self.valid_index = np.where(label != -1)[0]
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
        if len(bottom) != 3:
            raise Exception("Need 3 Inputs")

    def reshape(self,bottom,top):
        label = bottom[2].data
        # roi = bottom[1].data
        self.valid_index = np.where(label == -1)[0]
        self.count = len(self.valid_index)
        # print(roi[self.valid_index])
        # print(self.valid_index)
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
