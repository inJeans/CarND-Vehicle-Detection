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
from tqdm import tqdm

import numpy as np
import pandas as pd

IMAGE_FILETYPES = ["png", "PNG", 
                   "jpg", "JPG",
                   "jpeg", "JPEG"]

INPUT_DIR = "test_images"
OUTPUT_DIR = "output_images"

DATA_DIR = "data"
TRAIN_DIR = "train"
POSITIVE_LABELS = ["vehicles"]
NEGATIVE_LABELS = ["non-vehicles"]

TEST_SIZE = 0.2

COLOUR_SPACE = "YCrCb"
HOG_CHANNEL = "Grey"

class TrainClassifier(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    
    def output(self):
        output_path = os.path.join(self.output_directory, 
                                   "classifier.pkl")
        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return SplitData()

    def run(self):
        with self.input()[0].open("r") as train_file:
            train_filename = train_file.name
        train_df = pd.read_pickle(train_filename)

        with self.input()[1].open("r") as test_file:
            test_filename = test_file.name
        test_df = pd.read_pickle(test_filename)

        X_train, y_train = train_df.drop("label", axis=1).as_matrix(), train_df["label"].astype(np.int)
        X_test, y_test = test_df.drop("label", axis=1).as_matrix(), test_df["label"].astype(np.int)

        # Use a linear SVC (support vector classifier)
        svc = LinearSVC(dual=False,
                        verbose=True)
        # Train the SVC
        svc.fit(X_train, y_train)

        print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
        
        with self.output().open("wb") as output_file:
            joblib.dump(svc,
                        output_file)


class SplitData(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    test_size = luigi.parameter.FloatParameter(default=TEST_SIZE)
    
    def output(self):
        train_path = os.path.join(self.output_directory, 
                                  "train.pkl")
        test_path = os.path.join(self.output_directory, 
                                 "test.pkl")
        return [luigi.LocalTarget(train_path,
                                  format=luigi.format.MixedUnicodeBytesFormat(),
                                  is_tmp=False),
                luigi.LocalTarget(test_path,
                                  format=luigi.format.MixedUnicodeBytesFormat(),
                                  is_tmp=False)]

    def requires(self):
        return ScaleFeatures()

    def run(self):
        with self.input().open("r") as features_file:
            features_filename = features_file.name
        feature_df = pd.read_pickle(features_filename)

        test_df, train_df = train_test_split(feature_df,
                                             test_size=self.test_size)
        
        with self.output()[0].open("wb") as train_file:
            train_filename = train_file.name.split("-luigi-tmp")[0]
        train_df.to_pickle(train_filename)

        with self.output()[1].open("wb") as test_file:
            test_filename = test_file.name.split("-luigi-tmp")[0]
        test_df.to_pickle(test_filename)


class ScaleFeatures(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    
    def output(self):
        output_path = os.path.join(self.output_directory, 
                                   "scaled_features.pkl")
        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat(),
                                 is_tmp=False)

    def requires(self):
        return [FitScaler(), CombineFeatures()]

    def run(self):
        with self.input()[0].open("r") as scaler_pickle:
            scaler = joblib.load(scaler_pickle)
            
        with self.input()[1].open("r") as features_file:
            features_filename = features_file.name
        feature_df = pd.read_pickle(features_filename)

        scaled_features = scaler.transform(feature_df.drop("label",
                                                           axis=1).as_matrix())

        feature_df.loc[:, feature_df.columns != "label"] = scaled_features
        
        with self.output().open("wb") as output_file:
            output_filename = output_file.name.split("-luigi-tmp")[0]

        feature_df.to_pickle(output_filename)


class FitScaler(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    
    def output(self):
        output_path = os.path.join(self.output_directory, 
                                   "feature_scaler.pkl")
        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat())

    def requires(self):
        return CombineFeatures()

    def run(self):
        with self.input().open("r") as features_file:
            features_filename = features_file.name
        feature_df = pd.read_pickle(features_filename)

        scaler = StandardScaler().fit(feature_df.drop("label",
                                                      axis=1).as_matrix())
        
        with self.output().open("wb") as output_file:
            joblib.dump(scaler,
                        output_file)


class CombineFeatures(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    
    def output(self):
        output_name = "combined_features.pkl"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat(),
                                 is_tmp=False)

    def requires(self):
        return [
                GetHOGFeatures(),
                GetColourFeatures(),
                GetSpatialFeatures(),
                ]

    def run(self):
        combined_features_df = None
        for input_target in self.input():
            with input_target.open("r") as features_file:
                features_filename = features_file.name
        
            feature_df = pd.read_pickle(features_filename)
            if combined_features_df is None:
                combined_features_df = feature_df.copy()
            else:
                combined_features_df = combined_features_df.merge(feature_df.drop("label",
                                                                                  axis=1),
                                                                  left_index=True,
                                                                  right_index=True)

        print(combined_features_df)

        with self.output().open("wb") as output_file:
            output_filename = output_file.name.split("-luigi-tmp")[0]

        combined_features_df.to_pickle(output_filename)


class GetHOGFeatures(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    hog_channel = HOG_CHANNEL
    colour_space = COLOUR_SPACE
    
    def output(self):
        output_name = "hog_features.pkl"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat(),
                                 is_tmp=False)

    def requires(self):
        return LoadImages()

    def run(self):
        # Find out dataframe dimensions for memory prealloction
        number_instances = 0
        for l, input_target in enumerate(self.input()):
            with input_target.open("r") as input_file:
                image_array = np.load(input_file)
            number_instances += len(image_array)

        features = get_hog_features(image_array[0][:, :, 0],
                                    orient=9,
                                    pix_per_cell=8,
                                    cell_per_block=2,
                                    vis=False,
                                    feature_vec=True)
        number_features = len(features)
        if self.hog_channel == "ALL":
            number_features *= 3
        column_headers = ["hog-{}".format(f) for f in range(number_features)]
        column_headers.append("label")

        image_counter = 0
        features_array = np.zeros((number_instances, number_features+1))

        for l, input_target in enumerate(self.input()):
            with input_target.open("r") as input_file:
                print("{0} is {1}".format(input_file.name, l))
                image_array = np.load(input_file)
        
            for image in tqdm(image_array):
                image = convert_colour(image,
                                       colour_space=self.colour_space)
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(image.shape[2]):
                        hog_features.extend(get_hog_features(image[:, :, channel],
                                                             orient=9,
                                                             pix_per_cell=8,
                                                             cell_per_block=2,
                                                             vis=False,
                                                             feature_vec=True))
                elif self.hog_channel == "Grey":
                    grey_image = convert_colour(image,
                                                "grey")
                    hog_features = get_hog_features(grey_image,
                                                        orient=9,
                                                        pix_per_cell=8,
                                                        cell_per_block=2,
                                                        vis=False,
                                                        feature_vec=True)
                else:
                    hog_features = get_hog_features(image[:, :, self.hog_channel],
                                                        orient=9,
                                                        pix_per_cell=8,
                                                        cell_per_block=2,
                                                        vis=False,
                                                        feature_vec=True)
                features = np.array(hog_features)
                # features = np.concatenate(hog_features)
                features = np.append(features,
                                     l)
            
                features_array[image_counter, :] = features

                image_counter += 1

        hog_features_df = pd.DataFrame(data=features_array,
                                       index=range(number_instances),
                                       columns=column_headers)

        with self.output().open("wb") as output_file:
            output_filename = output_file.name.split("-luigi-tmp")[0]

        hog_features_df.to_pickle(output_filename)


def get_hog_features(image,
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     vis=False,
                     feature_vec=True):
    # feature_image = convert_colour(image,
    #                                colour_space="grey")
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
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    colour_space = COLOUR_SPACE
    
    def output(self):
        output_name = "colour_features.pkl"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat(),
                                 is_tmp=False)

    def requires(self):
        return LoadImages()

    def run(self):
        # Find out dataframe dimensions for memory prealloction
        number_instances = 0
        for l, input_target in enumerate(self.input()):
            with input_target.open("r") as input_file:
                print("{0} is {1}".format(input_file.name, l))
                image_array = np.load(input_file)
            number_instances += len(image_array)

        features = color_hist(image_array[0],
                              colour_space=self.colour_space,
                              nbins=32)
        number_features = len(features)
        column_headers = ["colour-{}".format(f) for f in range(number_features)]
        column_headers.append("label")

        image_counter = 0
        features_array = np.zeros((number_instances, number_features+1))

        for l, input_target in enumerate(self.input()):
            with input_target.open("r") as input_file:
                image_array = np.load(input_file)
        
            features_list = []
            for image in tqdm(image_array):
                features = color_hist(image,
                                      colour_space=self.colour_space,
                                      nbins=32)
                features = np.append(features,
                                     l)

                features_array[image_counter, :] = features

                image_counter += 1

        colour_features_df = pd.DataFrame(data=features_array,
                                          index=range(number_instances),
                                          columns=column_headers)

        with self.output().open("wb") as output_file:
            output_filename = output_file.name.split("-luigi-tmp")[0]

        colour_features_df.to_pickle(output_filename)


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


class GetSpatialFeatures(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)
    colour_space = COLOUR_SPACE

    def output(self):
        output_name = "spatial_features.pkl"
        output_path = os.path.join(self.output_directory, 
                                   output_name)

        return luigi.LocalTarget(output_path,
                                 format=luigi.format.MixedUnicodeBytesFormat(),
                                 is_tmp=False)

    def requires(self):
        return LoadImages()

    def run(self):
        # Find out dataframe dimensions for memory prealloction
        number_instances = 0
        for l, input_target in enumerate(self.input()):
            with input_target.open("r") as input_file:
                print("{0} is {1}".format(input_file.name, l))
                image_array = np.load(input_file)
            number_instances += len(image_array)

        features = bin_spatial(image_array[0],
                               colour_space=self.colour_space,
                               size=(32, 32))
        number_features = len(features)
        column_headers = ["spatial-{}".format(f) for f in range(number_features)]
        column_headers.append("label")

        image_counter = 0
        features_array = np.zeros((number_instances, number_features+1))

        for l, input_target in enumerate(self.input()):
            with input_target.open("r") as input_file:
                image_array = np.load(input_file)
        
            features_list = []
            for image in tqdm(image_array):
                features = bin_spatial(image,
                                       colour_space=self.colour_space,
                                       size=(32, 32))
                features = np.append(features,
                                     l)
                
                features_array[image_counter, :] = features

                image_counter += 1

        spatial_features_df = pd.DataFrame(data=features_array,
                                           index=range(number_instances),
                                           columns=column_headers)

        with self.output().open("wb") as output_file:
            output_filename = output_file.name.split("-luigi-tmp")[0]

        spatial_features_df.to_pickle(output_filename)


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
                   colour_space="RGB"):
    if colour_space == "RGB":
        new_image = np.copy(image)
    elif colour_space == 'HSV':
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
    elif colour_space == "grey":
        new_image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2GRAY)
    else:
        print("You did not enter a valid colour space")
        new_image = np.copy(image)

    return new_image


class LoadImages(luigi.Task):
    output_directory = luigi.Parameter(default=TRAIN_DIR)

    def output(self):
        output_file_list = []
        for task in self.requires():
            task_id = luigi.task.task_id_str(task.get_task_family(), 
                                             task.to_str_params())
            task_name = task_id.split("_")[0]
            output_path = os.path.join(self.output_directory, 
                                       "{}.npy".format(task_name))
            output_file_list.append(output_path)

        return [luigi.LocalTarget(output_path,
                                  format=luigi.format.MixedUnicodeBytesFormat())
                for output_path in output_file_list]

    def requires(self):
        return [PositiveTrainingImages(), NegativeTrainingImages()]

    def run(self):
        for input_list, output_target in zip(self.input(), self.output()):
            image_list = []

            for input_target in input_list:
                with input_target.open("r") as input_target_object:
                    image_filename = input_target_object.name

                image_np = cv2.imread(image_filename)
                image_list.append(image_np)

            image_array = np.array(image_list)

            with output_target.open("wb") as output_file:
                np.save(output_file,
                        image_array)


class NegativeTrainingImages(luigi.ExternalTask):
    image_dir = luigi.Parameter(default=DATA_DIR)
    label_list = luigi.parameter.ListParameter(default=NEGATIVE_LABELS)

    def output(self):
        image_list = []
        for label in self.label_list:
            label_path = os.path.join(self.image_dir,
                                      label)
            image_filenames = get_image_filenames(label_path)
            image_list.extend(image_filenames)

        return [luigi.LocalTarget(path=image_path) for image_path in image_list]


class PositiveTrainingImages(luigi.ExternalTask):
    image_dir = luigi.Parameter(default=DATA_DIR)
    label_list = luigi.parameter.ListParameter(default=POSITIVE_LABELS)

    def output(self):
        image_list = []
        for label in self.label_list:
            label_path = os.path.join(self.image_dir,
                                      label)
            image_filenames = get_image_filenames(label_path)
            image_list.extend(image_filenames)

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
