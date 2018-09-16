import numpy as np
import numpy.random as npr
size = 12
net = str(size)
img_dir = '/home/swang/work_space/MTCNN-Tensorflow/prepare_data/'

# with open('%s/pos_%s.txt'%(net, size), 'r') as f:
with open('/home/swang/work_space/MTCNN-Tensorflow/prepare_data/12/pos_12.txt', 'r') as f:
    pos2 = f.readlines()

# with open('%s/neg_%s.txt'%(net, size), 'r') as f:
with open('/home/swang/work_space/MTCNN-Tensorflow/prepare_data/12/neg_12.txt', 'r') as f:
    neg2 = f.readlines()

# with open('%s/part_%s.txt'%(net, size), 'r') as f:
with open('/home/swang/work_space/MTCNN-Tensorflow/prepare_data/12/part_12.txt', 'r') as f:
    part2 = f.readlines()

with open('/home/swang/work_space/MTCNN-Tensorflow/prepare_data/12/pts_12.txt', 'r') as f:
    pts2 = f.readlines()
    
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
    
import sys
import cv2
import os
import numpy as np
import pickle as pickle

all_list = []
print ('\n'+'positive-%d' %size)
cur_ = 0
sum_ = len(pos2)
pos_num = len(pos2)
for line in pos2:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = img_dir + words[0]
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/128
    label = 1
    roi = [float(words[2]), float(words[3]), float(words[4]), float(words[5])]
    pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    all_list.append([im,label,roi, pts])

print ('\n'+'negative-%d' %size)
cur_ = 0
if len(neg2) >= pos_num * 3:
    neg_keep = npr.choice(len(neg2), size=pos_num * 3, replace=False)
else:
    neg_keep = npr.choice(len(neg2), size=pos_num * 3, replace=True)
sum_ = len(neg_keep)
for i in neg_keep:
    line = neg2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = img_dir + words[0]
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/128
    label = 0
    roi = [-1,-1,-1,-1]
    pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    all_list.append([im,label,roi, pts])


print ('\n'+'part-%d' %size)
cur_ = 0
if len(part2) >= pos_num:
    part_keep = npr.choice(len(part2), size=pos_num, replace=False)
else:
    part_keep = npr.choice(len(part2), size=pos_num, replace=True)
sum_ = len(part_keep)
for i in part_keep:
    line = part2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = img_dir + words[0]
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5) / 128
    label = -1
    roi = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    all_list.append([im,label,roi, pts])


print ('\n'+'pts-%d' %size)
cur_ = 0
if len(pts2) >= pos_num:
    pts_keep = npr.choice(len(pts2), size=pos_num, replace=False)
else:
    pts_keep = npr.choice(len(pts2), size=pos_num, replace=True)
sum_ = len(pts_keep)
for i in pts_keep:
    line = pts2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = img_dir + words[0]
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5) / 128
    label = -2
    roi = [-1, -1, -1, -1]
    pts = [float(words[2]),float(words[3]),float(words[4]),float(words[5]), float(words[6]), float(words[7]),float(words[8]),float(words[9]),float(words[10]), float(words[11])]
    all_list.append([im,label,roi, pts])


fid = open("./%s/12_all.imdb" %size,'wb')
pickle.dump(all_list, fid)
fid.close()
