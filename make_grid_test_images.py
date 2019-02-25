import cv2
import numpy as np
from itertools import product
from os import path, listdir, mkdir

IMAGES_PATH = 'resources'
SUB_IMG_SHAPE = (509, 509)
OVERLAP = 0.25
STD_NOISE = 2
IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg', 'tif', 'tiff')


def make_grid_test_images(img_path):
    img = cv2.imread(img_path)
    new_path = path.join(path.dirname(img_path), 'grid_images', path.basename(img_path))
    mkdir(new_path)
    dx_step = SUB_IMG_SHAPE[1] - SUB_IMG_SHAPE[1] * OVERLAP
    dy_step = SUB_IMG_SHAPE[0] - SUB_IMG_SHAPE[0] * OVERLAP
    padded = np.zeros(
        np.ceil(np.ceil(np.array(img.shape) / (dy_step, dx_step, 1)) * (dy_step, dx_step, 1)).astype(int),
        dtype=np.uint8)
    padded[:img.shape[0], :img.shape[1], :] = img
    junk, shade = np.meshgrid(
        np.ones((SUB_IMG_SHAPE[0], )), np.linspace(255, 127, SUB_IMG_SHAPE[1]), indexing='ij')
    shade = np.dstack((shade,)*3).astype(np.uint8)
    dx = np.round(np.arange(0, padded.shape[1] - SUB_IMG_SHAPE[1], dx_step)).astype(int)
    dy = np.round(np.arange(0, padded.shape[0] - SUB_IMG_SHAPE[0], dy_step)).astype(int)
    print(f'Images per row: {len(dx)}')
    for idx, (dx, dy) in enumerate(product(dx, dy)):
        noise = np.random.normal(127, STD_NOISE, (*SUB_IMG_SHAPE, 3))
        cv2.imwrite(
                f'{new_path}/img_{idx}_x={dx}_y={dy}.png',
                cv2.addWeighted(
                    cv2.multiply(padded[dy:dy + SUB_IMG_SHAPE[0], dx:dx + SUB_IMG_SHAPE[1], :], shade, scale=1/255), 1,
                    np.round(noise).astype(np.uint8), 1, -127
                )
        )


def _is_image_path(file_path):
    return path.isfile(
        path.join(IMAGES_PATH, file_path)) and any(file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


if __name__ == '__main__':
    for img_file_path in filter(_is_image_path, listdir(IMAGES_PATH)):
        make_grid_test_images(path.join(IMAGES_PATH, img_file_path))
