import os

from os.path import isfile, join, splitext
from os import rename, listdir, rename, makedirs
from shutil import copyfile

source_folder_1 = '/scratch/ysolanki/FaceForensics++/original_sequences/youtube/c40/videos/'
source_folder_2 = '/scratch/ysolanki/FaceForensics++/manipulated_sequences/NeuralTextures/c40/videos/'
source_folder_3 = '/scratch/ysolanki/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos/'
source_folder_4 = '/scratch/ysolanki/FaceForensics++/manipulated_sequences/Face2Face/c40/videos/'
source_folder_5 = '/scratch/ysolanki/FaceForensics++/manipulated_sequences/FaceSwap/c40/videos/'
dest_folder_1 = '/scratch/ysolanki/FaceForensics++/train/1'
dest_folder_2 = '/scratch/ysolanki/FaceForensics++/train/0'
dest_folder_3 = '/scratch/ysolanki/FaceForensics++/test/1'
dest_folder_4 = '/scratch/ysolanki/FaceForensics++/test/0'




for i, j in zip(listdir(source_folder_1)[:860], listdir(source_folder_2)[:860]):
    name, ext = splitext(j)
    copyfile(join(source_folder_1, i), join(dest_folder_1, i))
    copyfile(join(source_folder_2, j), join(dest_folder_2, name + '_nt' + ext))
    copyfile(join(source_folder_3, j), join(dest_folder_2, name + '_df' + ext))
    copyfile(join(source_folder_4, j), join(dest_folder_2, name + '_ff' + ext))
    copyfile(join(source_folder_5, j), join(dest_folder_2, name + '_fs' + ext))

for i, j in zip(listdir(source_folder_1)[860:], listdir(source_folder_2)[860:]):
    name, ext = splitext(j)
    copyfile(join(source_folder_1, i), join(dest_folder_3, i))
    copyfile(join(source_folder_2, j), join(dest_folder_4, name + '_nt' + ext))
    copyfile(join(source_folder_3, j), join(dest_folder_4, name + '_df' + ext))
    copyfile(join(source_folder_4, j), join(dest_folder_4, name + '_ff' + ext))
    copyfile(join(source_folder_5, j), join(dest_folder_4, name + '_fs' + ext))


