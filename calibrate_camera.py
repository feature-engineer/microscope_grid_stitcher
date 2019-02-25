import cv2
import numpy as np


PATTERN_SIZE = (8, 11)


def save_camera_params(captured_img):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpoints = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objpoints[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)  # points corresponding to object chessboard.
    gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    if not ret:
        raise RuntimeError('Did not locate the corners as expected.')
    # refine corner positions.
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Draw and display the corners
    cv2.drawChessboardCorners(captured_img, PATTERN_SIZE, corners2, ret)
    cv2.imshow('img', captured_img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [corners2], gray.shape[::-1], None, None)
    h, w = captured_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(captured_img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult.png', dst)
    mean_error = 0
    img_points2, _ = cv2.projectPoints(objpoints, rvecs[0], tvecs[0], mtx, dist)
    error = cv2.norm(corners2, img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error
    print(f"total error: {mean_error}")
