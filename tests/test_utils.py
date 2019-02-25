import utils
import cv2
from itertools import product


def test_symmetric_matches_1_match_per_point():
    matches1 = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=10) for i, j in product(range(10), range(10))]
    matches2 = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=15) for i, j in product(range(5, 15), range(5, 15))]
    assert len(utils.symmetric_matches_1_match_per_point(matches1, matches2, 1)) == 0
    assert len(utils.symmetric_matches_1_match_per_point(matches1, matches2, 10)) == 25


def test_symmetric_matches_knn():
    matches1 = [
        [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=10) for i, j in product(range(k, k + 3), range(k, k + 3))]
        for k in range(10)
    ]
    matches2 = [
        [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=15) for i, j in product(range(k, k + 3), range(k, k + 3))]
        for k in range(5, 15)
    ]
    assert len(utils.symmetric_matches_k_matches_per_point(matches1, matches2, 1)) == 0
    symmetric = utils.symmetric_matches_k_matches_per_point(matches1, matches2, 10)
    assert len(symmetric) == 7
    assert len(symmetric[0]) == 1
    assert len(symmetric[1]) == 4
    assert len(symmetric[2]) == 9
    assert len(symmetric[-1]) == 9



