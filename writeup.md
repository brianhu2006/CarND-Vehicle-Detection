##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image21]: ./examples/HOG_example1.png
[image22]: ./examples/HOG_example2.png
[image3]: ./examples/sliding_windows.jpg
[image41]: ./output_images/find_cartest1.jpg
[image42]: ./output_images/find_cartest2.jpg
[image43]: ./output_images/find_cartest3.jpg
[image44]: ./output_images/find_cartest4.jpg
[image45]: ./output_images/find_cartest5.jpg
[image46]: ./output_images/find_cartest6.jpg

[image51]: ./output_images/heatmap_est1.jpg
[image52]: ./output_images/heatmap_test2.jpg
[image53]: ./output_images/heatmap_test3.jpg
[image54]: ./output_images/heatmap_test4.jpg
[image55]: ./output_images/heatmap_test5.jpg
[image56]: ./output_images/heatmap_test6.jpg


[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

```
def load_dataset():
    cars,cars_images=load_data('data/vehicles')
    notcars,notcars_images=load_data('data/non-vehicles')
    print(len(cars_images))
    print(len(notcars_images))
    return cars,cars_images,notcars,notcars_images


```

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
I use the following parametes:

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image21]
![alt text][image22]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and 	fianlly choose the following parameters

```
    colorspace='YCrCb'
    hog_channel = 'ALL'
    spatial = 32
    histbin = 32
    orient=9
    pix_per_cell = 8
    cell_per_block = 2
```
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using through the following steps:
 - read data of cars and notcars
 - extract color features and hog features
 - use standardscaler to standarize features
 - use linearSVC to train model
 - test svc
 - return svc and x_scaler

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

It does not work well and I choose to search the following area
```
Axis X start from 400
Axis Y start from 400-650
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image41] 
![alt text][image42] 
![alt text][image43]
![alt text][image44] 
![alt text][image45]
![alt text][image46]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/brianhu2006/CarND-Vehicle-Detection/blob/master/processed_video_project.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 


 I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here is example code.
```
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


def heat_map(image,dst_img, box_list):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, box_list)

    heat = apply_threshold(heat, 1)

    heatmap = np.clip(heat, 0, 255)
    #print(heatmap)

    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(dst_img, labels)
    
    return draw_img, heatmap

```
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
`
### Here are six frames and their corresponding heatmaps:

![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]
![alt text][image56]


### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image41] 
![alt text][image42] 
![alt text][image43]
![alt text][image44] 
![alt text][image45]
![alt text][image46]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, we use SVM to classify object in video and it works well to classify car and no-car only. In actually case, there are more objects in roads. We can  But it is not efficienct enough and in the future we can use faster-cnn to do object detction in future or next project.

