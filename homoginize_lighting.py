from skimage.restoration import inpaint
from skimage import feature
import cv2


def _remove_foreground(img):
    edges = feature.canny(img, sigma=3, low_threshold=10, high_threshold=200)

