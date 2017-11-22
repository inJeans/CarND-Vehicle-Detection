from __future__ import print_function

import luigi
import os
import glob
import cv2

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from collections import deque
from tqdm import tqdm

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

IMAGE_FILETYPES = ["png", "PNG", 
                   "jpg", "JPG",
                   "jpeg", "JPEG"]

INPUT_DIR = "test_images"
OUTPUT_DIR = "output_images"

DATA_DIR = "data"
TRAIN_DIR = "train"
POSITIVE_LABELS = ["vehicles"]
NEGATIVE_LABELS = ["non-vehicles"]

VIDEO = False
QUEUE_LENGTH = 20

TRAIN_WIDTH = 64
TRAIN_HEIGHT = 64

WINDOW_WIDTH = [32, 64, 96, 128]
WINDOW_HEIGHT = [32, 64, 96, 128]

# WINDOW_WIDTH = [96,]
# WINDOW_HEIGHT = [96,]

COLOUR_SPACE = "YCrCb"
HOG_CHANNEL = "Grey"
THRESHOLD = 5

MIN_WIDTH = 50
MAX_WIDTH = 300
MIN_HEIGHT = 50
MAX_HEIGHT = 300

MIN_ASPECT = 0.8
MAX_ASPECT = 2.75

BOX_COLOURS = [[255, 0, 0],
               [0, 255, 0],
               [0, 0, 255],
               [255, 255, 255],
               [0, 0, 0]]

class OverlayBoxes(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)

    def output(self):
        output_name = "video_output.mp4"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        if VIDEO:
            return [LoadVideo(), LabelHeatmap()]
        else:
            return [LoadTest(), LabelHeatmap()]

    def run(self):
        with self.input()[0].open("r") as input_file:
            image_array = np.load(input_file)
        with self.input()[1].open("r") as input_file:
            box_array = np.load(input_file)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("bounding_boxes.mp4",fourcc, 20.0, (1280,720))

        for image, boxes in zip(tqdm(image_array), box_array):
            car_count = 0
            for b, box in enumerate(boxes):
                aspect_is_okay = False
                size_is_okay = False
                box_min = (box[0][0], box[0][1])
                box_max = (box[1][0], box[1][1])
                width = box_max[0] - box_min[0]
                height = box_max[1] - box_min[1]
                aspect = width / height
                if width < MAX_WIDTH and width > MIN_WIDTH and height < MAX_HEIGHT and height > MIN_HEIGHT:
                    size_is_okay = True
                if aspect < MAX_ASPECT and aspect > MIN_ASPECT:
                    aspect_is_okay = True
                if size_is_okay and aspect_is_okay:
                    car_count += 1
                    # Draw the box on the image
                    cv2.rectangle(image, box_min, box_max, BOX_COLOURS[0], 3)

            out.write(image)
        
        out.release()

class BoundingBoxes(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)

    def output(self):
        output_name = "bounding_boxes.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return LabelHeatmap()

    def run(self):
        with self.input().open("r") as input_file:
            label_array = np.load(input_file)

        image_list = []
        for labels in tqdm(label_array):
            box_list = get_bboxes(labels)

            image_list.append(box_list)
        
        image_array = np.array(image_list)

        with self.output().open("wb") as output_file:
            np.save(output_file,
                    image_array)

def get_bboxes(labels):
    box_list = []
    # Iterate through all detected cars
    print(labels[1])
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        box_list.append(bbox)

    return np.array(box_list)

class LabelHeatmap(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)
    threshold = THRESHOLD

    def output(self):
        output_name = "labeled_heatmap.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return [LoadTest, ThresholdHeatmap()]

    def run(self):
        with self.input()[0].open("r") as input_file:
            image_array = np.load(input_file)
        with self.input()[1].open("r") as input_file:
            heatmap_array = np.load(input_file)

        label_list = []
        image_list = []
        for heatmap in tqdm(heatmap_array):
            labels = label(heatmap)
            print(labels[1], 'cars found')

            box_list = get_bboxes(labels)
            image_list.append(box_list)
        
            label_list.append(labels)
        
        image_array = np.array(image_list)
        label_array = np.array(label_list)
        
        with self.output().open("wb") as output_file:
            np.save(output_file,
                    image_array)

class ThresholdHeatmap(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)
    threshold = THRESHOLD

    def output(self):
        output_name = "thresholded_heatmap.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return GetHeatmap()

    def run(self):
        with self.input().open("r") as input_file:
            heatmap_array = np.load(input_file)

        thresholded_list = []
        if VIDEO:
            thresholded_queue = deque(maxlen=QUEUE_LENGTH)
        for heatmap in tqdm(heatmap_array):
            if VIDEO:
                thresholded_queue.append(heatmap)
            thresholded_heatmap = np.zeros_like(heatmap)

            if VIDEO:
                hot_indices = np.mean(thresholded_queue, axis=0) > self.threshold
            else:
                hot_indices = heatmap > self.threshold

            thresholded_heatmap[hot_indices] = 1

            thresholded_list.append(thresholded_heatmap)
        
        thresholded_array = np.array(thresholded_list)
        with self.output().open("wb") as output_file:
            np.save(output_file,
                    thresholded_array)

class GetHeatmap(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)

    def output(self):
        output_name = "heatmap.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        if VIDEO:
            return [LoadVideo(), GetHotWindows()]
        else:
            return [LoadTest(), GetHotWindows()]

    def run(self):
        with self.input()[0].open("r") as input_file:
            image_array = np.load(input_file)
        with self.input()[1].open("r") as input_file:
            window_array = np.load(input_file)

        heatmap_list = []
        for image, window_list in zip(tqdm(image_array), window_array):
            image = convert_colour(image, "RGB")
            heatmap = np.zeros((image.shape[0], image.shape[1]))
            for window in window_list:
                heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

            heatmap_list.append(heatmap)
        
        heatmap_array = np.array(heatmap_list)
        with self.output().open("wb") as output_file:
            np.save(output_file,
                    heatmap_array)

class GetHotWindows(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)
    colour_space = COLOUR_SPACE
    hog_channel = HOG_CHANNEL

    def output(self):
        output_name = "hot_windows.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        if VIDEO:
            return [LoadVideo(),
                    GetWindows(),
                    Classifier(),
                    Scaler()]
        else:
            return [LoadTest(),
                    GetWindows(),
                    Classifier(),
                    Scaler()]
        

    def run(self):
        with self.input()[0].open("r") as input_file:
            image_array = np.load(input_file)

        with self.input()[1].open("r") as input_file:
            window_array = np.load(input_file)

        with self.input()[2].open("r") as classifier_pickle:
            classifier = joblib.load(classifier_pickle)

        with self.input()[3].open("r") as scaler_pickle:
            scaler = joblib.load(scaler_pickle)
        
        hot_windows_list = []
        for image in tqdm(image_array):
            hot_windows = search_windows(image,
                                              window_array,
                                              classifier,
                                              scaler,
                                              color_space=self.colour_space, 
                                              spatial_size=(32, 32),
                                              hist_bins=32,
                                              orient=9,
                                              pix_per_cell=8,
                                              cell_per_block=2,
                                              hog_channel=self.hog_channel)

            hot_windows_list.append(hot_windows)

        hot_windows_array = np.array(hot_windows_list)

        
        with self.output().open("wb") as output_file:
            np.save(output_file,
                    hot_windows_array)

def search_windows(img,
                   windows,
                   clf,
                   scaler,
                   color_space='RGB', 
                   spatial_size=(32, 32),
                   hist_bins=32,
                   orient=9,
                   pix_per_cell=8,
                   cell_per_block=2,
                   hog_channel="ALL"):

    #1) Create an empty list to receive positive detection windows
    hot_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        height = window[1][1] - window[0][1] 
        width = window[1][0] - window[0][0]
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                              (TRAIN_WIDTH, TRAIN_HEIGHT))
        features = get_single_image_features(test_img,
                                             color_space=color_space,
                                             spatial_size=spatial_size,
                                             hist_bins=hist_bins,
                                             orient=orient, 
                                             pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block,
                                             hog_channel=hog_channel,
                                             spatial_feat=True,
                                             hist_feat=True,
                                             hog_feat=True)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 0:
            hot_windows.append(window)
    #8) Return windows for positive detections
    return hot_windows

def get_single_image_features(image,
                              color_space='RGB',
                              spatial_size=(32, 32),
                              hist_bins=32,
                              orient=9, 
                              pix_per_cell=8,
                              cell_per_block=2,
                              hog_channel="ALL",
                              spatial_feat=True,
                              hist_feat=True,
                              hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []  
    #7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(image.shape[2]):
                hog_features.extend(get_hog_features(image[:,:,channel],
                                                     orient=orient,
                                                     pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block,
                                                     vis=False,
                                                     feature_vec=True))  
        elif hog_channel == "Grey":
            grey_image = convert_colour(image,
                                        "grey")
            hog_features = get_hog_features(grey_image,
                                            orient=orient,
                                            pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            vis=False,
                                            feature_vec=True) 
        else:
            hog_features = get_hog_features(image[:, :, hog_channel],
                                            orient=orient,
                                            pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            vis=False,
                                            feature_vec=True)
    #8) Append features to list
    img_features.append(hog_features)
    #5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(image,
                                   colour_space=color_space,
                                   nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(image,
                                       colour_space=color_space,
                                       size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

class GetWindows(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)

    def output(self):
        output_name = "windows.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        if VIDEO:
            return LoadVideo()
        else:
            return LoadTest()

    def run(self):
        with self.input().open("r") as input_file:
            image_array = np.load(input_file)

        window_list = []
        for width, height in zip(tqdm(WINDOW_WIDTH), WINDOW_HEIGHT):
            windows = slide_window(image_array[0],
                                   y_start_stop=[400, 656], 
                                   xy_window=(width, height),
                                   xy_overlap=(0.5, 0.5))
            window_list.extend(windows)

        window_array = np.array(window_list)
        # window_array = np.array(windows)
        
        with self.output().open("wb") as output_file:
            np.save(output_file,
                    window_array)

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None], 
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


class GetHOGFeatures(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)
    colour_space = "RGB"
    
    def output(self):
        output_name = "test_hog_features.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return LoadTest()

    def run(self):
        # Find out dataframe dimensions for memory prealloction
        with self.input().open("r") as input_file:
            image_array = np.load(input_file)
        number_instances = len(image_array)

        features = get_hog_features(image_array[0],
                                    orient=9,
                                    pix_per_cell=8,
                                    cell_per_block=2,
                                    vis=False,
                                    feature_vec=True)
        number_features = len(features)

        features_array = np.zeros((number_instances, number_features))

        with self.input().open("r") as input_file:
            image_array = np.load(input_file)
        
        for i, image in enumerate(tqdm(image_array)):
            features = get_hog_features(image,
                                        orient=9,
                                        pix_per_cell=8,
                                        cell_per_block=2,
                                        vis=False,
                                        feature_vec=True)
            features_array[i, :] = features

        with self.output().open("wb") as output_file:
            np.save(output_file,
                    features_array)

def get_hog_features(image,
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     vis=False,
                     feature_vec=True):
    if len(image.shape) > 2:
        if image.shape[2] > 1:
            image = convert_colour(image,
                                   colour_space="grey")
            # feature_image = np.copy(image)
    return_list = None    
    if vis == True:
        features, hog_image = hog(image,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False, 
                                  visualise=True,
                                  feature_vector=False,
                                  block_norm="L2-Hys")
        return_list = features, hog_image
    else:      
        features = hog(image,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False, 
                       visualise=False,
                       feature_vector=feature_vec,
                       block_norm="L2-Hys")
        return_list = features

    return return_list


class GetColourFeatures(luigi.Task):
    output_directory = luigi.Parameter(default=OUTPUT_DIR)
    colour_space = "RGB"
    
    def output(self):
        output_name = "test_colour_features.npy"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return LoadTest()

    def run(self):
        # Find out dataframe dimensions for memory prealloction
        with self.input().open("r") as input_file:
            image_array = np.load(input_file)
        number_instances = len(image_array)

        features = color_hist(image_array[0],
                              colour_space=self.colour_space,
                              nbins=32)
        number_features = len(features)

        features_array = np.zeros((number_instances, number_features))

        with self.input().open("r") as input_file:
            image_array = np.load(input_file)
        
        features_list = []
        for i, image in enumerate(tqdm(image_array)):
            features = color_hist(image,
                                  colour_space=self.colour_space,
                                  nbins=32)
            features_array[i, :] = features

        with self.output().open("wb") as output_file:
            np.save(output_file,
                    features_array)


def color_hist(image,
               colour_space="RGB",
               nbins=32):
    if colour_space != 'RGB':
        feature_image = convert_colour(image,
                                       colour_space=colour_space)
    else: 
        feature_image = np.copy(image)

    # Assumes three colour channels
    channel1_hist = np.histogram(feature_image[:,:,0],
                                 bins=nbins)
    channel2_hist = np.histogram(feature_image[:,:,1],
                                 bins=nbins)
    channel3_hist = np.histogram(feature_image[:,:,2],
                                 bins=nbins)

    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))

    return hist_features


def bin_spatial(image,
                colour_space='RGB',
                size=(32, 32)):
    # Convert image to new color space (if specified)
    if colour_space != 'RGB':
        feature_image = convert_colour(image,
                                       colour_space=colour_space)
    else: 
        feature_image = np.copy(image)             
    
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()

    # Return the feature vector
    return features


def convert_colour(image,
                   colour_space="HSV"):
    if colour_space == 'HSV':
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2HSV)
    elif colour_space == 'LUV':
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2LUV)
    elif colour_space == 'HLS':
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2HLS)
    elif colour_space == 'YUV':
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2YUV)
    elif colour_space == 'YCrCb':
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2YCrCb)
    elif colour_space == "RGB":
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2RGB)
    elif colour_space == "grey":
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2GRAY)
    else:
        print("You did not enter a valid colour space")
        new_image = np.copy(image)

    return new_image


class LoadVideo(luigi.Task):
    output_dir = luigi.Parameter(default=OUTPUT_DIR)

    def output(self):
        output_path = os.path.join(self.output_dir,
                                   "test_video.npy")

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return TestVideo()

    def run(self):
        with self.input().open("r") as input_target_object:
            video_filename = input_target_object.name
    
        cap = cv2.VideoCapture(video_filename)
        ret = True

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        depth = 3

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(width, height, depth, num_frames)

        frame_array = np.empty((num_frames, height, width, depth),
                               dtype=np.uint8)
        for frame in tqdm(range(num_frames)):
            _, captured_frame = cap.read()
            try:
                frame_array[frame] = captured_frame.astype(np.uint8)
            except AttributeError:
                print("It looks like we had a problem with frame {}".format(frame))

        with self.output().open("wb") as output_file:
            np.save(output_file,
                    frame_array)

class LoadTest(luigi.Task):
    output_dir = luigi.Parameter(default=OUTPUT_DIR)

    def output(self):
        output_path = os.path.join(self.output_dir,
                                   "test_images.npy")

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return TestImages()

    def run(self):
        image_list = []
        for input_target in self.input():
            with input_target.open("r") as input_target_object:
                image_filename = input_target_object.name
            print(image_filename)
            image_np = cv2.imread(image_filename)
            image_list.append(image_np)

        image_array = np.array(image_list)

        with self.output().open("wb") as output_file:
            np.save(output_file,
                    image_array)


class Classifier(luigi.ExternalTask):
    train_dir = luigi.Parameter(default=TRAIN_DIR)

    def output(self):
        classifier_path = os.path.join(self.train_dir,
                                       "classifier.pkl")

        return luigi.LocalTarget(path=classifier_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())


class Scaler(luigi.ExternalTask):
    train_dir = luigi.Parameter(default=TRAIN_DIR)

    def output(self):
        scaler_path = os.path.join(self.train_dir,
                                   "feature_scaler.pkl")

        return luigi.LocalTarget(path=scaler_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())


class TestVideo(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(path="project_video.mp4")

class TestImages(luigi.ExternalTask):
    image_dir = luigi.Parameter(default=INPUT_DIR)

    def output(self):
        image_list = get_image_filenames(self.image_dir)

        return [luigi.LocalTarget(path=image_path) for image_path in image_list]


def get_image_filenames(image_dir):
    image_filename_list = []

    sub_directory_list = glob.glob(image_dir + "/*/")

    if len(sub_directory_list) > 0:
        for sub_directory in glob.glob(image_dir + "/*/"):
            image_filenames = get_image_filenames(sub_directory)
            if len(image_filenames) > 0:
                image_filename_list.extend(image_filenames)
    else:
        for filetype in IMAGE_FILETYPES:
            image_filenames = glob.glob(image_dir + "/*." + filetype)
            if len(image_filenames) > 0:
                image_filename_list.extend(image_filenames)

    return image_filename_list
