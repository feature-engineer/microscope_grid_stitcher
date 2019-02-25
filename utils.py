

def symmetric_matches_1_match_per_point(matches1, matches2, tolerance):
    match2_map = {(m.trainIdx, m.queryIdx): m.distance for m in matches2}
    symmetric_matches = []
    for match in matches1:
        match2_distance = match2_map.get((match.queryIdx, match.trainIdx))
        if match2_distance and abs(match.distance - match2_distance) < tolerance:
            symmetric_matches.append(match)
    return symmetric_matches


def symmetric_matches_k_matches_per_point(matches1, matches2, tolerance):
    match2_map = {
        (m.trainIdx, m.queryIdx): m.distance for k_matches in matches2 for m in k_matches}
    total_symmetric_matches = []
    for k_matches in matches1:
        symmetric_matches = []
        for match in k_matches:
            match2_distance = match2_map.get((match.queryIdx, match.trainIdx))
            if match2_distance and abs(match.distance - match2_distance) < tolerance:
                symmetric_matches.append(match)
        if symmetric_matches:
            total_symmetric_matches.append(symmetric_matches)
    return total_symmetric_matches


def distinct_matches(matches):
    good_matches = []
    for bm1, bm2 in matches:
        if bm1.distance < 0.7 * bm2.distance:
            good_matches.append(bm1)
    return good_matches
