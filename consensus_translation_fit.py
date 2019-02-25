import numpy as np


def consensus_translation_fit(matches, query_points, train_points, tolerance, min_inleirs):
    best_inliers = []
    for match_idx, k_candidates in enumerate(matches):
        for candidate in k_candidates:
            dxdy = query_points[candidate.queryIdx] - train_points[candidate.trainIdx]
            inliers = []
            for other_match_idx, other_k_candidates in enumerate(matches):
                if other_match_idx == match_idx:
                    continue
                best_candidate = _best_fit_for_translation(dxdy, other_k_candidates, query_points, train_points)
                if np.sum(
                    np.square(query_points[best_candidate.queryIdx] - train_points[best_candidate.trainIdx] - dxdy), 1
                ) < tolerance:
                    inliers.append((query_points[best_candidate.queryIdx] - train_points[best_candidate.trainIdx]))
            if len(inliers) > min_inleirs and len(inliers) > len(best_inliers):
                best_inliers = inliers
    best_inliers = np.row_stack(best_inliers)
    if not best_inliers:
        return None, None
    return np.mean(best_inliers, 0), np.std(best_inliers, 0)


def _best_fit_for_translation(dxdy, k_candidates, query_points, train_points):
    return k_candidates[np.argmin(
        np.sum(
            np.square(query_points[k_candidates.queryIdx] - train_points[k_candidates.trainIdx] - dxdy), 1
        ))]
