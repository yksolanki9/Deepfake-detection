# !pip install facenet_pytorch

# !pip install scikit-image

# !pip install opencv-python

# !apt install libgl1-mesa-glx

from facenet_pytorch import MTCNN
import cv2
from PIL import Image

from os import listdir, makedirs
import glob
from os.path import join, exists
from skimage.io import imsave
import imageio.core.util
import time
import torch


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# +
# Create face detector
# If you want to change the default size of image saved from 160, you can
# uncomment the second line and set the parameter accordingly.
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda"
)

# mtcnn = torch.nn.DataParallel(mtcnn, device_ids=list(range(torch.cuda.device_count())))
# mtcnn = MTCNN(margin=40, select_largest=False, post_process=False,
# device='cuda:0', image_size=256)
# -

# Directory containing images respective to each video
source_frames_folders = ["/scratch/ysolanki/SiameseAkash/deepfakes_video_classification/train_frames/1"]
# Destination location where faces cropped out from images will be saved
dest_frames_folders = ["/scratch/ysolanki/SiameseAkash/deepfakes_video_classification/train_face/1"]


for i, d in zip(source_frames_folders, dest_frames_folders):
    counter = 0
    start = time.time()
    for j in listdir(i):
        imgs = glob.glob(join(i, j, "*.jpg"))
        if counter % 10 == 0:
            end = time.time()
            print("Number of videos done:", counter, " and time taken is ", (end-start)/60, " mins")
            start = time.time()
        if exists(join(d, j)):
            counter += 1
            continue
        else:
            makedirs(join(d, j))
        for k in imgs:
            frame = cv2.imread(k)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            try:
                face = mtcnn(frame)
            except TypeError:
                print("No image found in ", j ,"th video")
        
            else:
                try:
                    imsave(
                        join(d, j, k.split("/")[-1]),
                        face.permute(1, 2, 0).int().numpy(),
                        check_contrast=False
                    )
                except AttributeError:
                    print("Image skipping")
        counter += 1

torch.cuda.device_count()


