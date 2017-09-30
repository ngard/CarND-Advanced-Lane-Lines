import os, cv2
import numpy as np

# Define Max Curve Radius
kMaxCurveR = 3000
# Define conversions in x and y from pixels space to meters
kMeterPerPixelY = 10./160 # meters per pixel in y dimension
kMeterPerPixelX = 3.7/700 # meters per pixel in x dimension

kWeightPrevCycle = 1000

def read_images(image_dir):
    fname_images = os.listdir(image_dir)
    for fname_image in fname_images:
        if not fname_image.endswith(".jpg"):
            continue
        img = cv2.imread(image_dir+fname_image)
        yield img, fname_image

def read_movie(movie_path):
    cap = cv2.VideoCapture(movie_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True:
            break
        yield frame
    cap.release()

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def undistort_image(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    return cv2.undistort(img, mtx, dist, None, mtx)
        
def calc_transform_matrix(src,dst):
    return cv2.getPerspectiveTransform(src,dst)

def warp_image(img,M):
    return cv2.warpPerspective(img,M,img.shape[1::-1])

def grayscale_image(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def saturation_image(img):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    return hls[:,:,2] # returns only saturation

def detect_edge(img_gray):
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    return np.absolute(sobelx)

def binary_white(img_warped):
    threashold_min = 150
    white_binary = np.zeros_like(img_warped[:,:,0])
    white_binary[(img_warped[:,:,0] >= threashold_min) &
                 (img_warped[:,:,1] >= threashold_min) &
                 (img_warped[:,:,2] >= threashold_min)] = 255
    return white_binary

def binary_yellow(img_warped):
    threashold = 255/2
    yellow_binary = np.zeros_like(img_warped[:,:,0])
    yellow_binary[(img_warped[:,:,0] <= threashold) &
                  (img_warped[:,:,1] >= threashold) &
                  (img_warped[:,:,2] >= threashold)] = 255
    return yellow_binary

def binary_white_or_yellow(img_warped):
    yellow_binary = binary_yellow(img_warped)
    white_binary = binary_white(img_warped)
    yw_binary = np.zeros_like(img_warped[:,:,0])
    yw_binary[(yellow_binary==255) | (white_binary==255)] = 255
    return yw_binary

def mix_images(img_edge, img_saturation, img_binary_color):
    assert img_edge.size == img_saturation.size
    assert img_edge.size == img_binary_color.size

    height_img = img_edge.shape[0]
    for row in range(height_img):
        img_edge[row] = img_edge[row] * (height_img-row)

    img_normalized_edge = np.zeros_like(img_edge)
    img_normalized_saturation = np.zeros_like(img_edge)
    
    img_normalized_edge = cv2.normalize(img_edge, img_normalized_edge, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    img_normalized_saturation = cv2.normalize(img_saturation, img_normalized_saturation,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8U)

    return np.dstack((img_binary_color/10, img_normalized_edge, img_normalized_saturation))

def combine_images(img_edge, img_saturation, img_binary_color):
    assert img_edge.size == img_saturation.size
    assert img_edge.size == img_binary_color.size

    img_mix = mix_images(img_edge, img_saturation, img_binary_color)
    img_combined = np.zeros_like(img_binary_color)
    img_combined = img_mix[:,:,0]*img_mix[:,:,1]*img_mix[:,:,2]
    #ret, img_combined = cv2.threshold(img_combined,30,255,cv2.THRESH_TOZERO)
    return img_combined

def undistort_and_warp_test_images(M):
    generator_image = read_images("../test_images/")
    for (img, fname) in generator_image:
        img_undist = undistort_image(img,mtx,dist)
        cv2.imshow("",img_undist)
        cv2.waitKey(200)
        img_warped = warp_image(img_undist,M)
        cv2.imwrite("../test_images/warped/"+fname,img_warped)
        cv2.imshow("",img_warped)
        cv2.waitKey(500)

def undistort_and_warp_and_detect_edge_on_movie(M,movie_path=""):
    for img_original in read_movie(movie_path):
        img_undist = undistort_image(img_original,mtx,dist)
        img_warped = warp_image(img_undist,M)

        img_warped_gray = grayscale_image(img_warped)
        img_warped_edge = detect_edge(img_warped_gray)
        img_warped_saturation = saturation_image(img_warped)
        img_binary_white_or_yellow = binary_white_or_yellow(img_warped)
        
        img_mix = mix_images(img_warped_edge,img_warped_saturation,img_binary_white_or_yellow)
        img_combined = combine_images(img_warped_edge,img_warped_saturation,img_binary_white_or_yellow)
        yield img_original, img_warped, img_mix, img_combined
        
def make_sliding_window(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 10
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                      (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                      (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img, left_fit, right_fit

def fit_line(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))&
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    leftw = []
    rightw = []
    for (x,y) in zip(leftx,lefty):        
        leftw.append(binary_warped[y,x])
    for (x,y) in zip(rightx,righty):
        rightw.append(binary_warped[y,x])

    # Plot Points from Previous Cycle Result
    height_img = binary_warped.shape[0]
    ploty = np.linspace(0, height_img-1, height_img)
    left_fitx_prev = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx_prev = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    left_weight_prev = np.ones_like(left_fitx_prev)*kWeightPrevCycle
    right_weight_prev = np.ones_like(right_fitx_prev)*kWeightPrevCycle
    
    leftx = np.hstack((leftx,left_fitx_prev))
    rightx = np.hstack((rightx,right_fitx_prev))
    lefty = np.hstack((lefty,ploty))
    righty = np.hstack((righty,ploty))
    leftw = np.hstack((leftw,left_weight_prev))
    rightw = np.hstack((rightw,right_weight_prev))

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2, w=leftw)
    right_fit = np.polyfit(righty, rightx, 2, w=rightw)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    left_line_pts = np.array(zip(left_fitx, ploty), np.int32)
    right_line_pts = np.array(zip(right_fitx, ploty), np.int32)
    cv2.polylines(window_img, [left_line_pts], False, (255,255,255),1)
    cv2.polylines(window_img, [right_line_pts], False, (255,255,255),1)

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result, left_fit, right_fit

def calc_curvature(left_fit, right_fit, y_eval):    
    # Fit new polynomials to x,y in world space 
    ploty = np.linspace(0, y_eval-1, y_eval )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_fit_cr = np.polyfit(ploty*kMeterPerPixelY, leftx*kMeterPerPixelX, 2)
    right_fit_cr = np.polyfit(ploty*kMeterPerPixelY, rightx*kMeterPerPixelX, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*kMeterPerPixelY + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*kMeterPerPixelY + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    left_curverad = kMaxCurveR if left_curverad > kMaxCurveR else left_curverad
    right_curverad = kMaxCurveR if right_curverad > kMaxCurveR else right_curverad

    # Curve to Left gets negative radius and Curve to Right gets positive radius
    left_curverad = left_curverad if left_fit_cr[0] < 0 else -left_curverad
    right_curverad = right_curverad if right_fit_cr[0] < 0 else -right_curverad
    
    return left_curverad, right_curverad

def calc_offset(left_fit, right_fit, shape_img):
    height_img = shape_img[0]
    width_img = shape_img[1]
    offset_left = left_fit[0]*height_img**2 + left_fit[1]*height_img + left_fit[2]
    offset_right = right_fit[0]*height_img**2 + right_fit[1]*height_img + right_fit[2]
    offset_center = ((offset_left+offset_right)/2 - width_img/2)*kMeterPerPixelX
    return offset_center

def overlay_curvature(img_original, left_curverad, right_curverad, offset_center, confidence):
    assert type(confidence) == bool
    is_straight = False
    #curverad = (left_curverad+right_curverad)/2
    curverad_inv = ((1./left_curverad + 1./right_curverad)/2)
    if (curverad_inv <= 1./kMaxCurveR and curverad_inv >= -1./kMaxCurveR):
        is_straight = True
    if not is_straight:
        curverad = 1./curverad_inv
    text_curvature = "Curvature:  Straight" if (is_straight) else "Curvature:%5.0f[m] "%(abs(curverad))
    if not is_straight:
        text_curvature += "(Right)" if curverad < 0 else "(Left)"

    text_color = (0,0,0)
    cv2.putText(img_original,text_curvature,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,text_color)
    text_offset = "Offset:%4.2f[m] "%abs(offset_center)
    text_offset += "(Right)" if offset_center < 0 else "(Left)"
    cv2.putText(img_original,text_offset,(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,text_color)

def overlay_lane(img_original, Minv, left_fit, right_fit):
    y_eval = img_original.shape[0]
    ploty = np.linspace(0, y_eval-1, y_eval)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    
    # Create an image to draw the lines on
    warped_fill_lane = np.zeros_like(img_original).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warped_fill_lane, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_fill_lane = cv2.warpPerspective(warped_fill_lane, Minv, img_original.shape[1::-1])
    # Combine the result with the original image
    return cv2.addWeighted(img_original, 1, img_fill_lane, 0.3, 0)

def show_lane_on_movie(M,Minv,movie_path):
    is_first = True
    left_fit = 0
    right_fit = 0
    for img_original, img_warped, img_mix, img_combined in undistort_and_warp_and_detect_edge_on_movie(M,movie_path):
        cv2.imshow("original",img_original)
        cv2.imshow("warped",img_warped)
        cv2.imshow("mix",img_mix)
        if is_first:
            img, left_fit, right_fit = make_sliding_window(img_combined)
            cv2.imshow("sw",img)
            is_first = False
        else:
            img, left_fit, right_fit = fit_line(img_combined, left_fit, right_fit)
            cv2.imshow("sw",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        left_R, right_R= calc_curvature(left_fit,right_fit,img.shape[0])
        offset_center = calc_offset(left_fit,right_fit,img.shape)
        overlay_curvature(img_original,left_R,right_R,offset_center,True)
        img_overlay = overlay_lane(img_original,Minv,left_fit,right_fit)
        cv2.imshow("overlay",img_overlay)
        
mtx = np.load("matrix.npy")
dist = np.load("distortion_coeffs.npy")
print("Camera Matrix",mtx)
print("Distortion Coeffs",dist)

src = np.float32([[598,450],[685,450],[280,680],[1050,680]])
dst = np.float32([[280,0],[1000,0],[280,720],[1000,720]])
M = calc_transform_matrix(src,dst)
Minv = calc_transform_matrix(dst,src)

video = "../project_video.mp4"
#video = "../challenge_video.mp4"
#video = "../harder_challenge_video.mp4"

show_lane_on_movie(M,Minv,video)

cv2.destroyAllWindows()
