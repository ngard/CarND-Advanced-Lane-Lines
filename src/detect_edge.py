import os, cv2
import numpy as np

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
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img,M,img_size)

def grayscale_image(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def saturation_image(img):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    return hls[:,:,2] # returns only saturation

def detect_edge(img_gray):
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    return np.absolute(sobelx)

def binary_edge(img_edge):
    threashold_min = 20
    sxbinary = np.zeros_like(img_edge)
    sxbinary[img_edge >= threashold_min] = 255
    return sxbinary

def binary_saturation(img_saturation):
    threashold_min = 170
    threashold_max = 500
    s_binary = np.zeros_like(img_saturation)
    s_binary[(img_saturation >= threashold_min) & (img_saturation <= threashold_max)] = 255
    return s_binary

def mix_binary_images(img_binary_edge, img_binary_saturation):
    return np.dstack((np.zeros_like(img_binary_saturation), img_binary_edge, img_binary_saturation))

def undistort_test_images():
    generator_image = read_images("../test_images/")
    target_dir = "../test_images/undistorted/"
    mkdir_if_not_exists(target_dir)
    for (img, fname) in generator_image:
        cv2.imshow("",img)
        cv2.waitKey(200)
        img_undist = undistort_image(img,mtx,dist)
        cv2.imwrite(target_dir+fname,img_undist)
        cv2.imshow("",img_undist)
        cv2.waitKey(400)

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

def undistort_and_warp_and_detect_edge_on_test_images(M):
    generator_image = read_images("../test_images/")
    target_dir_edge = "../test_images/edge/"
    target_dir_saturation = "../test_images/saturation/"
    target_dir_mix = "../test_images/mix/"
    mkdir_if_not_exists(target_dir_edge)
    mkdir_if_not_exists(target_dir_saturation)
    mkdir_if_not_exists(target_dir_mix)
    for (img, fname) in generator_image:
        img_undist = undistort_image(img,mtx,dist)
        img_warped = warp_image(img_undist,M)
        img_warped_gray = grayscale_image(img_warped)
        img_warped_edge = detect_edge(img_warped_gray)
        img_binary_edge = binary_edge(img_warped_edge)
        cv2.imwrite(target_dir_edge+fname,img_warped_edge)
        
        img_warped_saturation = saturation_image(img_warped)
        img_binary_saturation = binary_saturation(img_warped_saturation)
        cv2.imwrite(target_dir_saturation+fname,img_binary_saturation)

        img_mix = mix_binary_images(img_binary_edge,img_binary_saturation)
        cv2.imwrite(target_dir_mix+fname,img_mix)

def undistort_and_warp_and_detect_edge_on_movie(M,movie_path=""):
    for img in read_movie(movie_path):
        cv2.imshow("original",img)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_undist = undistort_image(img,mtx,dist)
        img_warped = warp_image(img_undist,M)
        img_warped_gray = grayscale_image(img_warped)
        img_warped_edge = detect_edge(img_warped_gray)
        img_binary_edge = binary_edge(img_warped_edge)
        
        img_warped_saturation = saturation_image(img_warped)
        img_binary_saturation = binary_saturation(img_warped_saturation)

        img_mix = mix_binary_images(img_binary_edge,img_binary_saturation)
        cv2.imshow("edge",img_mix)
        cv2.waitKey(1)
        
mtx = np.load("matrix.npy")
dist = np.load("distortion_coeffs.npy")
print(mtx,dist)

#undistort_test_images()

src = np.float32([[598,450],[685,450],[280,680],[1050,680]])
dst = np.float32([[300,0],[1000,0],[300,720],[1000,720]])
M = calc_transform_matrix(src,dst)

#undistort_and_warp_test_images(M)

undistort_and_warp_and_detect_edge_on_test_images(M)

#undistort_and_warp_and_detect_edge_on_movie(M,"../project_video.mp4")

cv2.destroyAllWindows()
