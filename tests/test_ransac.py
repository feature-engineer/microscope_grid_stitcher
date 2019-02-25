from ransac import ransac
import translation_model
import numpy as np


def test_ransac():
    model = translation_model.TranslationModel()
    points1 = np.array([np.array([i, i]) for i in range(10)])
    points2 = np.array([np.array([i, i]) for i in range(10, 20)])
    points2[:3] = np.array([np.array([i, i]) for i in range(3)])
    data = np.column_stack((points1, points2))
    fit_model = ransac(data, model, 2, 100, 1, 4, False)
    assert fit_model == (-10, -10)
