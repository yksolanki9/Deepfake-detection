# Deepfake-detection
Deepfake detection using Triplet Loss

SampleDatasetFromElvin - A subset containing 1000 real and fake images from dataset stored at "/raid/Data/Master_Dataset_Elvin_Mrunal/Facebook_created_dataset_Face_extracted_Real+Fake_Small/train"

SampleDatasetFromElvin/resize - SampleDatasetFromElvin contains extracted faces of various size. This folder contains those images resized to 160x160

SampleDatasetFromElvin/resize/sample-dataset-from-elvin-1000rf.npz - File containing input(2000, 160, 160, 3) and output(2000, 1)

SampleDatasetFromElvin/resize/sample-dataset-elvin-1000rf-faces-embeddings.npz - Generated face embeddings using FaceNet

deepfakes_video_classification - Clone of "https://github.com/AKASH2907/deepfakes_video_classification" and tested for DGX

YashSiamese - Code to generate "SampleDatasetFromElvin"
