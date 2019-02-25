from consensus_translation_fit import consensus_translation_fit
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import utils


IMAGES_PATH = 'images'
IMAGES_PER_ROW = 5

image_paths = sorted([f for f in listdir(IMAGES_PATH) if isfile(join(IMAGES_PATH, f)) and f.endswith('png')])

descriptor_builder = cv2.ORB_create()
# use BFMatcher for binary feature descriptors such as ORB, FLANN for floats
# NORM_HAMMING is the distance used for ORB descriptors
matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

for idx, img_path in enumerate(image_paths):
    row = idx // IMAGES_PER_ROW
    col = idx % IMAGES_PER_ROW
    img = cv2.imread(img_path)
    if idx == 0:
        img_height = img.shape[0]
        img_width = img.shape[1]
        prev_img = img[:, 3 * img_width / 4:, :].copy()
        img_mask = 255 * np.ones(img.shape[:2], dtype=np.uint8)
        img_mask[3 * img_height / 4:, 3 * img_width / 4:] = 0
        continue
    img_points, img_descriptors = descriptor_builder.computeAndDetect(img, img_mask)
    other_img_points, other_img_descriptors = descriptor_builder.computeAndDetect(prev_img, None)
    if row > 0:
        img_above = cv2.imread(image_paths[idx + IMAGES_PER_ROW])[3 * img_height / 4, :, :]
        img_above_points, img_above_descriptors = descriptor_builder.computeAndDetect(img_above, None)
        other_img_points.extend(img_above_points)
        other_img_descriptors.extend(img_above_descriptors)
    matches1 = matcher.knnMatch(img_descriptors, other_img_descriptors, 5)
    matches2 = matcher.knnMatch(img_descriptors, other_img_descriptors, 5)
    matches = utils.symmetric_matches_k_matches_per_point(matches1, matches2, 5)
    (dx, dy), (dx_std, dy_std) = consensus_translation_fit(
        matches, cv2.KeyPoint.convert(img_points), cv2.KeyPoint.convert(other_img_points), 9, 20)

    print(f'offest = ({dx}±{dx_std}, {dy}±{dy_std})')



