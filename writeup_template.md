## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
I will use this template. Thank you :-)

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[heat1]: ./output_images/heatmap-01.png
[heat2]: ./output_images/heatmap-02.png
[heat3]: ./output_images/heatmap-03.png
[heat4]: ./output_images/heatmap-04.png
[heat5]: ./output_images/heatmap-05.png
[heat6]: ./output_images/heatmap-06.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! I'm using it!

### Breif Overview
Just before we get started, I just wanted to give a breif overview of the program structure and the Python package I used to build the pipeline. The code has been broken up into two pipeline scripts; one for training (`classifier_training_pipeline.py`) and one for detecting (`vehicle_detection_pipeline.py`). To construct the pipeline I have used a Python package developed by Spotify, [Luigi](https://luigi.readthedocs.io/en/stable/). It is a development tool for piecing together machine learning pipelines. It is great for things that don't require immediate results, but not so great if you want real time feedback. So while it's use may have been suitable for this project it is certainly not appropriate for a real self driving car.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 252 through 285 of the file called `classifier_training_pipeline`.  

I started by reading in all the `vehicle` and `non-vehicle` images into memory and saving them into a single numpy array.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and just investigated which combination gave the bets bounding boxes once I had finished the entire pipeline. While using the validation accuracy was a useful guide it doesn't capture the aesthetic value of the bounding boxes. It may have been better to use some MAP metric on the actual bounding boxes to better gage the performance of the model in an algorithmic way.

One thing that seemed to give the best performance was to use a single greyscale image for the HOG feature extraction. I tried combining all three colour channels using various different colour spaces, but a single greyscale image seemed t work the best. I guess this makes sense as the HOG extractor is really looking for structural elements which don't necessarily depend on colour.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Lines 56 - 63 of `classifier_training_pipeline.py` are where I trained a linear SVM using a vector composed of HOG features, colour bins and spatial features. I just used the `train` method of `scikit-learn`'s `LinerSVC` class, leaving the default parameters as is. It was very straight forward.

As always it is the data prep that consumes the majority of the pipeline. Here I extracted the three different types of features, concatenated them together, scaled them using `scikit-learn`'s `StandardScaler` class and then split them in an 80-20 train-test split using another built in `sk-learn` function.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I ended up using a range of window sizes to try to capture vehicle at different scales. After playing with the test images I noticed that 128x128 was probably about the largest size I would need, while 32x32 was certainly the smallest. Then I added in a few more window size in between for good measure.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using greyscale HOG features plus spatially binned color (YCrCb) and histograms of color (YCrCb) in the feature vector, which provided a nice result.  

#### Here are six frames with their heatmaps overlayed:

![alt text][heat1]
![alt text][heat2]
![alt text][heat3]
![alt text][heat4]
![alt text][heat5]
![alt text][heat6]

These heatmaps looks pretty bad, but they were improved by averaging across many frames as discussed in the next section.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/jwaZsSvvLms)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap. This heatmap was then averaged over the last `N` frames to ensure that an object consistently appeared in a similar location. Then I thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

One improvement to this technique may be to look at IoU values for boxes in subsequent frames and only keep the ones with a large IoU value. This would ensure that boxes are similar to those seen in the previous frame.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I made a few silly mistake early on which resulted in some poor results. Generally they were problems associated with the data munging, things like concatenating arrays together, making sure labels were correctly assigned etc. 

From the results I've seen so far it looks like this pipeline is triggered by small dark object, so as with the earlier projects shadows may be a problem. I have implemented some frame smoothing which has ameliorated the problem somewhat, but it is definitely still there to some extent. The smoothing is executed by simply keeping a list of the last `N` heatmaps and averaging over them. We can then threshold the image in the same way we did for the single frame processing, it just means that an object needs to be persistently detected between frames for it to register.

I think some kind of center tracking or object counting could increase the robustness of the method. You could predict future bounding box centers based of the velocities and positions of the current centers and then see whether your new predictions approximately match the detected ones. This might also help with smoothing.

