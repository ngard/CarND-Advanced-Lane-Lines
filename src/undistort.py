import os
import cv2
import numpy as np

nx = 6#TODO: enter the number of inside corners in x
ny = 9#TODO: enter the number of inside corners in y

def read_images():
    image_dir = "../camera_cal/"
    fname_images = os.listdir(image_dir)
    for fname_image in fname_images:
        img = cv2.imread(image_dir+fname_image)
        yield img, fname_image
        
def find_corners(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    return corners.reshape(-1,2) if ret == True else None

def draw_corners(img,corners):
    return cv2.drawChessboardCorners(img, (nx, ny), corners, True)

def calc_camera_matrix(img, imgpoints):
    # Generate object points corresponded to nx and ny
    objpoints = []
    for points in imgpoints:
        objpoints.append(np.zeros((nx*ny,3), np.float32))
        objpoints[-1][:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    return mtx,dist

def undistort(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    return cv2.undistort(img, mtx, dist, None, mtx)

generator_image_1 = read_images()
generator_image_2 = read_images()
imgpoints = []

# Here, this program reads images of chessboard patterns,
# finds corners and saves them to imgpoints.
for img,fname in generator_image_1:
    corners = find_corners(img)
    if corners is None:
        continue
    imgpoints.append(corners)
    img = draw_corners(img,corners)
    cv2.imshow("",img)
    cv2.waitKey(200)

# The line below calculates Camera Matrix and Distortion Coeffs
# from the imgpoints which have been found above.
mtx,dist = calc_camera_matrix(img,imgpoints)
print("Camera Matrix",mtx)
print("Distortion Coeffs",dist)

# The lines below undistorts the chessboard patters
# with Camera Maxrix and Distortion Coeffs which have been calcurated above.
# Then shows and saves them.
for img, fname in generator_image_2:
    cv2.imshow("",img)
    cv2.waitKey(100)
    img_undist = undistort(img,mtx,dist)
    cv2.imshow("",img_undist)
    cv2.imwrite("../output_images/undistort/"+fname,img_undist)
    cv2.waitKey(200)
cv2.destroyAllWindows()

# The lines below save calcurated Camera Matrix and Distortion Coeffs to npy file
# to use them in other programs.
np.save("matrix.npy",mtx)
np.save("distortion_coeffs.npy",dist)
