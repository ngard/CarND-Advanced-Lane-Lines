## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration3.jpg "Original Chessboard Pattern"
[image1]: ./output_images/undistort/calibration3.jpg "Undistorted Chessboard Pattern"
[image2]: ./output_images/lane/00019_undistort.png "Undistorted Image"
[image31]: ./output_images/straight_lines1_undistorted.jpg "Road Undistorted"
[image32]: ./output_images/straight_lines1_warped.jpg "Road Transformed"
[image4]: ./output_images/lane/00019_mix.png "Filtered Image"
[image5]: ./output_images/lane/00019_line.png "Fit Line"
[image6]: ./output_images/lane/00019_overlay.png "Output"
[video1]: ./output_images/lane_overlay.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #44 through #80 of the file called `src/undistort.py`).  

First, I searched the corners on given chessboard patterns using cv2.findChessboardCorners().

Then, I calculated the camera matrix and distortion coeffs with cv2.calibrateCamera().
As the arguments of the function, I had to prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  

By using calcurated Camera Matrix and Distortion Coeffs,
I undistort the given images with cv2.undistort() function.
The result is shown as below:

![Original Chessboard Pattern][image0]

![Undistorted Chessboard Pattern][image1]

### Pipeline (single images)

For this project, I made a pipeline which does not followed the instruction.

At the first step, I perspective-transformed the images to birds-eye view. Then, I applied filters to detect lane markers.

Therefore, my steps does not meet to the Project Rublic at some points (i.e. Requirements No.3 comes before No.2), however, I finally succeeded to find lanes.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![An example of undistorted image][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is realized by 2 steps.

First, I calcurated M, Transform Matrix, using cv2.warpPerspectiveTransform() in `src/detect_lane.py:37 calc_transform_matrix()`.
The function takes two inputs, source (`src`) and destination (`dst`) points to calcurate the matrix.  I chose the hardcode the source and destination points as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 598, 450      |  280,   0     | 
| 685, 450      | 1000,   0     |
| 280, 680      |  280, 720     |
|1050, 680      | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![An example of undistort image][image31]

![An example of transformed image][image32]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For this project, I did not use "binary" image to detect lanes out of images in order to utilize filtered value of each pixel as weight for fitting polynomial lines.
I used a combination of color (both Saturation and RGB) and gradient thresholds to generate a filtered image (thresholding steps at lines #119 through #125 in `src/detect_lane.py`).  Here's an example of my output for this step. (R shows the strength of Saturation, G shows the strength of Edge and B shows each pixel has white or yellow color.)

![Filtered Image of Lane Lines][image4]

Finally, the product of the 3 channels above (Saturation, Edge and Color) is used to fit polynomial lines.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I followed the instructions to fit polynomial lines on to filetered images as using sliding windows to find out lines on the first image and then use look-ahead filter (from #378 to #385 in `src/detect_lane.py`).

For fitting line, I used the value of each pixel in filtered image (`img_combined` in source code) as weight because the bigger the value the more likely the pixel represents the lane.

This works good in some confusing scene when it is hard to distinguish lane markers as below.

![Result of Polynomial Line Fitting to Filtered Image][image5]

I also implemented low-path-filter by adding points of previous cycle when fitting line (from #235 to #241 in `src/detect_lane.py`) to suppress fluctuation.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calcurated the curvature and offset within `calc_curvature()` and `calc_offset()` in `src/detect_lane.py`.

It mostly follows the instruction, however, it uses the sign of curvature as the direction of curvature (left or right) and it limits the outputs upto 3000[m] because the value more than that is not so reliable and the image just looks like straight road then.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #389 through #390 in my code in `src/detect_line.py` in the function `overlay_curvature_and_offset()` and `overlay_lane()`.  Here is an example of my result on a test image:

![Result of Lane Detection][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/lane_overlay.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Once this program fails to detect lane, it never recover since there is no sanity-check or re-initialization feature in the code. Therefore, by implementing these features this program becomes more robust.

Also, the parameters in the code (like threasholds or weights) is not fine-tuned enough. They are needed to be elaborated.