import cv2
import numpy as np
import glob

imm = cv2.imread('images/img1.png', cv2.IMREAD_COLOR)
imm_rgb = cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)
cv2.imshow('img1', imm_rgb)
cv2.waitKey(0)

# convert to gray scale
imm_gray = cv2.cvtColor(imm_rgb, cv2.COLOR_RGB2GRAY)

# Chessboard size
pattern = (24, 17)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
ret, corners = cv2.findChessboardCorners(imm_gray, pattern)

if ret:
    acc_corners = cv2.cornerSubPix(imm_gray, corners, (12, 12), (-1, -1), criteria)
    cv2.drawChessboardCorners(imm_rgb, pattern, acc_corners, ret)
    imm_conv = cv2.resize(imm_rgb, (1080, 720))
    cv2.imshow('img-corners', imm_conv)
    print(imm_gray.shape[::-1])
    cv2.waitKey(0)


objpoints = []
imgpoints = []

objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
objp[ :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

objpoints.append(objp)
imgpoints.append(acc_corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imm_gray.shape, None, None)
print(f'k1: {dist[0][0]}, k2: {dist[0][1]}, p1: {dist[0][2]}, p2: {dist[0][3]}, k3: {dist[0][4]}')

img = cv2.imread('images/img5.png', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img_rgb, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imshow('img5_with_undistort_1', dst)
cv2.waitKey(0)

# calibrate with img1 to img4
new_objpoints = []
new_imgpoints = []
images = glob.glob('images/*.png')
curr_im = None
for fname in images[:-1]:
    print(fname)
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    curr_im = gray
    ret, corners = cv2.findChessboardCorners(gray, pattern)
    if ret:
        new_objpoints.append(objp)
        acc_corners = cv2.cornerSubPix(gray, corners, (12, 12), (-1, -1), criteria)
        new_imgpoints.append(acc_corners)
        cv2.drawChessboardCorners(img, pattern, acc_corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)


img = cv2.imread('images/img5.png', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(new_objpoints, new_imgpoints, curr_im.shape, None, None)
h, w = img_rgb.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undistort
dst = cv2.undistort(img_rgb, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imshow('img5_with_undistort_all', dst)
cv2.waitKey(0)
