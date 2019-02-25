import translation_model
import numpy as np


def test_translation_model():
    model = translation_model.TranslationModel()
    points1 = np.array([np.array([i, i]) for i in range(10)])
    points2 = np.array([np.array([i, i]) for i in range(10, 20)])
    data = np.column_stack((points1, points2))
    train_data = data[5:]
    test_data = data[:5]
    fit_model = model.fit(train_data)
    error = model.get_error(test_data, fit_model)
    assert fit_model == (-10, -10)
    assert error < 0.00001
    points2[:3] = np.array([np.array([i, i]) for i in range(3)])
    data = np.column_stack((points1, points2))
    train_data = data[5:]
    test_data = data[:5]
    fit_model = model.fit(train_data)
    error = model.get_error(test_data, fit_model)
    assert error == 6
