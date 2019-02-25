import calibrate_camera
import numpy as np
import cv2
from itertools import product

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
NUM_VERTICAL_SQUARES = 9
NUM_HORIZONTAL_SQUARES = 6
SQUARE_SIZE = min(IMG_HEIGHT // (NUM_VERTICAL_SQUARES * 2), IMG_WIDTH // (NUM_HORIZONTAL_SQUARES * 2))
HORIZONTAL_BORDER = (IMG_WIDTH - NUM_HORIZONTAL_SQUARES * SQUARE_SIZE * 2) // 2
VERTICAL_BORDER = (IMG_HEIGHT - NUM_VERTICAL_SQUARES * SQUARE_SIZE * 2) // 2 + SQUARE_SIZE // 2


def _get_checker_board():
    checker_board = 255 * np.ones((IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)
    for i, j in product(range(NUM_HORIZONTAL_SQUARES), range(NUM_VERTICAL_SQUARES)):
        horizontal_corner_position = i * 2 * SQUARE_SIZE + HORIZONTAL_BORDER + (j % 2) * SQUARE_SIZE
        vertical_corner_position = j * SQUARE_SIZE + VERTICAL_BORDER
        checker_board[
        horizontal_corner_position:horizontal_corner_position + SQUARE_SIZE,
        vertical_corner_position:vertical_corner_position + SQUARE_SIZE
        ] = np.zeros((SQUARE_SIZE, SQUARE_SIZE), dtype=np.uint8)
    return checker_board


def test_save_camera_params():
    # checker_board = cv2.imread('/home/user/PycharmProjects/grid_stitch/tests/146958763448279.jpg')
    checker_board = np.dstack([_get_checker_board(), _get_checker_board(), _get_checker_board()])
    RT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 10]])
    f_to_pixel_size = 10
    K = np.array([[1, 0, IMG_WIDTH / 2], [0, 1, IMG_HEIGHT / 2], [0, 0, 2]])
    #K = np.array([[f_to_pixel_size, 0, IMG_WIDTH / 2], [0, f_to_pixel_size, IMG_HEIGHT / 2], [0, 0, 1]])
    #cv2.projectPoints(
    #    np.mgrid[:8, :6].T.reshape(-1, 2), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  np.array([0, 0, 0]), K, None)
    #warped = cv2.warpPerspective(chekcer_board, K, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imwrite('/home/user/PycharmProjects/grid_stitch/tests/dist.png', checker_board)
    calibrate_camera.save_camera_params(checker_board)

