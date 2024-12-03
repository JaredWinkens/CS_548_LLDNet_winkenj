# This is a fork created for a CS 548 class project at SUNY Polytechnic Institute

**Team Members:** Jared Winkens

## Setup

1. Clone this repo
```
git clone https://github.com/JaredWinkens/CS_548_LLDNet_winkenj.git
```

2. Create a virtual environment with `Python 3.6.13` and install the required packages.
    - `Keras 2.6.0`
    - `OpenCV 4.10.0.84`
    - `Tensorflow 2.10.1`
    - `Sklearn 1.5.2`
    - `Pandas 2.2.3`
    - `Numpy 1.26.4`
    - `Matplotlib 3.9.2`

3. Download the training dataset images and labels from the links below
    
    ***Make sure to add them to your local repo where the heirarchy is as follows: `CS_548_LLDNET_WINKENJ/data/images_mixed.npy` & `CS_548_LLDNET_WINKENJ/data/labels_mixed.npy`***

    Images: https://drive.google.com/file/d/1S23Ac0_hbOktV0rE2q0IkQWpQjUfkMTB/view?usp=sharing 
    
    Labels: https://drive.google.com/file/d/1I264WVBL3Dyp_4PTfEYkVIDkg_Yn5gJJ/view?usp=sharing

4. Download the test videos from the link below
    
    ***Make sure to add the folder to the root directory of your local repo: `CS_548_LLDNET_WINKENJ/vids/`***
    
    Test videos: https://drive.google.com/drive/folders/12P33jY8AijRAunk7o-nMXzxVoN4xd5v9?usp=sharing

## Experiment 1

1. Follow steps **1 - 4** in **Setup** above (if not already done)

2. To get the quantitative results of the experiment, run each of the steps in `LLDNet.ipynb`
    
    ***Make sure to change any hardcoded paths to your own paths***

3. To get the qualitative results of the experiment:
    - Open `Lane_detection.py`
    - Change the path in `video_capture = cv2.VideoCapture("vids/input/vid_0.mp4")` to whatever video you want to test.
    - Run `Lane_detection.py`
    - You can find the ouput video in `CS_548_LLDNET_WINKENJ/vids/predicted/j.avi`

## Experiment 2

1. Follow steps **1 - 4** in **Setup** above (if not already done)

2. To get the quantitative results of the experiment, run each of the steps in `LLDNet_Experiment_2.ipynb`
    
    ***Make sure to change any hardcoded paths to your own paths***

3. To get the qualitative results of the experiment:
    - Open `Lane_detection.py`
    - Change the path in `model = load_model('LLDNet.h5', ...)` to `'LLDNet_Experiment_2.h5'`
    - Change the path in `video_capture = cv2.VideoCapture("vids/input/vid_0.mp4")` to whatever video you want to test.
    - Run `Lane_detection.py`
    - You can find the ouput video in `CS_548_LLDNET_WINKENJ/vids/predicted/j.avi`

# LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning

This is the source code of LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning. We provide the dataset, model development code, test code and the pretrained model.

Khan, M.A.-M.; Haque, M.F.; Hasan, K.R.; Alamjani, S.H.; Baz, M.; Masud, M.; Al-Nahid, A. LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning. Sensors 2022, 22, 5595. https://doi.org/10.3390/s22155595

# Network Architecture
![image](https://user-images.githubusercontent.com/33350185/181172871-77060790-4437-44c9-ba26-d88ee953e114.png)

# Some Results
![image](https://user-images.githubusercontent.com/33350185/181173323-d740d99e-29f1-4bad-946c-57e10f17db11.png)
![image](https://user-images.githubusercontent.com/33350185/181174138-31ac678a-080f-44eb-9ad0-2b0dac4efcb7.png)
![image](https://user-images.githubusercontent.com/33350185/181174183-7788e669-e02e-4215-b771-9319c78a31fe.png)

# Dataset
## Description 
In this work, we constructed a mixed dataset by combining the “Udacity Machine Learning Nanodegree Project Dataset” and the “Cracks and Potholes in Road Images Dataset”. The first dataset was collected by a smartphone and contained 12,764 training images. The images were extracted from 12 different videos filmed at 30 fps. The speciality of this dataset is that it contains images of different weather conditions, different road curvatures, and different lighting conditions. The other dataset of our work collected images by using a Highway Diagnostic Vehicle (DHV). This dataset contains 2235 images which were extracted from a few videos filmed at 30 fps with. The speciality of this dataset is that most of the images in this dataset contain images with cracks and holes on the road. However, the roads on the images are not heavily damaged; rather, the images contain roads with minor damage. Each image of this dataset contains three mask images, including lane marking, hole marking, and crack marking masks. However, we only consider the lane marking masks images of this dataset. The primary purpose of mixing these two datasets in our work is that we want to develop a robust system. The system can detect the lanes in adverse weather or lighting conditions, even if the roads are defective and unstructured.

## Data Preparation
The original images and the labeled images of the first dataset were converted into two different pickle files and uploaded to Mr. Michael Vigro’s GitHub page (https://github.com/mvirgo/MLND-Capstone). After downloading the pickle files, this work converted them into NumPy files. The second dataset was downloded from https://data.mendeley.com/datasets/t576ydh9v8/4. After downloading, we resized the second dataset in the same size (80×160×3) as the first dataset for mixing the two datasets. After that, the original and labeled images were imported as NumPy arrays and stored in two different Numpy files. And these two NumPy files were merged with the Numpy files of the previous dataset for generating a mixed dataset.

## Download
The mixed dataset utilized in this work can be found https://drive.google.com/file/d/1S23Ac0_hbOktV0rE2q0IkQWpQjUfkMTB/view?usp=sharing 
and https://drive.google.com/file/d/1I264WVBL3Dyp_4PTfEYkVIDkg_Yn5gJJ/view?usp=sharing

# Key Files
1. [LLDNet.ipynb](https://github.com/Masrur02/LLDNet/blob/main/LLDNet.ipynb)- Assuming you have downloaded the training images and labels above, this is the proposed LLDNet to train using that data.

2. [LLDNet.h5](https://github.com/Masrur02/LLDNet/blob/main/LLDNet.h5)-These are the final outputs from the above CNN. Note that if you train the file above the originals here will be overwritten! These get fed into the below.
3. [Lane_detection.py](https://github.com/Masrur02/LLDNet/blob/main/Lane_detection.py) -Using the trained model and an input video, this predicts the lane, and returns the original video with predicted lane lines drawn onto it.

# Citation
Please cite our paper if you use this code or the mixed dataset in your own work:\
@article{sensors-22-05595},\
  title={LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning},\
  author={Khan, M.A.-M.; Haque, M.F.; Hasan, K.R.; Alamjani, S.H.; Baz, M.; Masud, M.; Al-Nahid, A.},\
  journal={Sensors},\
  volume={22},\
  year={2022}\
}



