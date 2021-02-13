# !pip install pandas

import pandas as pd
import glob
import os
from os.path import isfile, join, split
from os import rename, listdir, rename, makedirs
from random import shuffle
# from sklearn.utils import shuffle

abs_path = "/scratch/ysolanki/SiameseAkash/deepfakes_video_classification/train_face/"
train_path = ["/scratch/ysolanki/SiameseAkash/deepfakes_video_classification/train_face/1", "/scratch/ysolanki/SiameseAkash/deepfakes_video_classification/train_face/0"]

list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]
list_0 = [join(train_path[1], x) for x in listdir(train_path[1])]

c = 0

vid_list = list_1 + list_0
print(len(vid_list))
shuffle(vid_list)

images = []
labels = []

counter = 0

for x in vid_list:
    img = glob.glob(join(abs_path, x, '*.jpg'))
    img.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
    images+=img[:25]
    label = [k.split('/')[-3] for k in img]
    labels+=label[:25]

    if counter%100==0:
        print("Number of files done:", counter)
    counter+=1

# print(images)
# print(labels)

data = {
	'images_list': images,
	'label': labels
	}

df = pd.DataFrame(data)
df.to_csv("train_faces_25frames.csv", index=False)


import pandas as pd
data = pd.read_csv("../train_face/train_faces_25frames.csv")
print(data.loc[data['label'] == 1])

# + active=""
#
# -

# test_path = ['train_face_160/0', 'train_face_160/1']
# list_1 = [join(test_path[0], x) for x in listdir(test_path[0])]
# list_0 = [join(test_path[1], x) for x in listdir(test_path[1])]

# list_all = list_1 + list_0
# shuffle(list_all)

# labels = []

# for i in list_all:
# 	label = i.split('/')[1]
# 	labels+=label

# data = {
# 	'vids_list': list_all,
# 	'label': labels
# }

# df = pd.DataFrame(data)
# df.to_csv('train_faces_160.csv', index=False)
