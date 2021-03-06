import numpy as np
import numpy.random as npr
size = 12
net = str(size)
with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos2 = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg2 = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part2 = f.readlines()
    
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


roi_list = []
print ('\n'+'part-%d' %size)
cur_ = 0
part_keep = npr.choice(len(part2), size=300000, replace=False)
sum_ = len(part_keep)
for i in part_keep:
    line = part2[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = './' + words[0] + '.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5) / 127.5
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])

print ('\n'+'positive-%d' %size)
cur_ = 0
sum_ = len(pos2)
for line in pos2:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = './' + words[0] + '.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=size or w!=size:
        im = cv2.resize(im,(size,size))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    # label    = -1
    label = 1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])


fid = open("./%s/roi.imdb" %size, 'wb')
pickle.dump(roi_list, fid)
fid.close()
