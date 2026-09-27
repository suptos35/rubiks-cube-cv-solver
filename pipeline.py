"""
Rubik's Cube Image-to-Solution Pipeline — debuggable reference implementation.
Adapted to expose intermediate contour stages, compactness scores, and detailed validation.
"""

import sys
import itertools
import cv2
import numpy as np
import kociemba

# ----------------------------------------------------------------------------
# Default tunable parameters
# ----------------------------------------------------------------------------
DEFAULT_SATURATION_THRESHOLD = 40
DEFAULT_GLOBAL_HSV_LOWER_V   = 80     # colored-pixel brightness floor
DEFAULT_WHITE_VALUE_THRESHOLD = 200   # raised from 160 — fixes paper/cube merge bug
DEFAULT_MIN_CONTOUR_AREA     = 100
DEFAULT_RATIO_THRESHOLD      = 0.6    # rectangularity: contour area / min-rect area
DEFAULT_CANDIDATE_LIMIT      = 15
DEFAULT_COMPACTNESS_THRESHOLD = 0.70
DEFAULT_SPREAD_TOLERANCE_MULT = 1.5
DST_SIZE                     = 300
FACE_ORDER                   = ['u', 'r', 'f', 'd', 'l', 'b']
FACE_LABELS                  = {'u': 'U', 'r': 'R', 'f': 'F', 'd': 'D', 'l': 'L', 'b': 'B'}


# ----------------------------------------------------------------------------
# Stage 1 — Segmentation
# ----------------------------------------------------------------------------
def get_masks(img, saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
              lower_v=DEFAULT_GLOBAL_HSV_LOWER_V,
              white_v=DEFAULT_WHITE_VALUE_THRESHOLD):
    """
    Generate the color mask, white mask, and morphological cleaned mask.
    Returns: (color_mask, white_mask, cleaned_mask)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(
        hsv,
        np.array([0, saturation_threshold, lower_v]),
        np.array([179, 255, 255]),
    )
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, white_v]),
        np.array([179, max(0, saturation_threshold - 1), 255]),
    )
    sticker_mask = cv2.bitwise_or(color_mask, white_mask)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(sticker_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return color_mask, white_mask, cleaned


def segment_stickers(img, saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
                     lower_v=DEFAULT_GLOBAL_HSV_LOWER_V,
                     white_v=DEFAULT_WHITE_VALUE_THRESHOLD):
    """Returns only the cleaned binary mask for sticker segmentation."""
    _, _, cleaned = get_masks(img, saturation_threshold, lower_v, white_v)
    return cleaned


# ----------------------------------------------------------------------------
# Stage 2 — Contour extraction and candidate filtering
# ----------------------------------------------------------------------------
def find_all_contours(mask):
    """Find and return all raw contours without filtering."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def filter_rectangular_candidates(contours, min_area=DEFAULT_MIN_CONTOUR_AREA,
                                  ratio_threshold=DEFAULT_RATIO_THRESHOLD):
    """
    Filter contours by minimum area and minAreaRect rectangularity ratio.
    Returns: list of survivor contours.
    """
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        box_area = rect[1][0] * rect[1][1]
        if box_area <= 0:
            continue
        if (area / box_area) >= ratio_threshold:
            candidates.append(c)
    return candidates


def find_rectangular_candidates(mask, min_area=DEFAULT_MIN_CONTOUR_AREA,
                                ratio_threshold=DEFAULT_RATIO_THRESHOLD):
    """
    Convenience function returning both raw contours and filtered candidates.
    Returns: (raw_contours, filtered_candidates)
    """
    raw_contours = find_all_contours(mask)
    filtered = filter_rectangular_candidates(raw_contours, min_area, ratio_threshold)
    return raw_contours, filtered


# ----------------------------------------------------------------------------
# Stage 3 — Select exactly 9 stickers
# ----------------------------------------------------------------------------
def calculate_compactness_score(contours_group):
    """
    Compute compactness: sum(contour areas) / minAreaRect(all points).
    Returns a float score (0.0 to 1.0+).
    """
    if not contours_group or len(contours_group) != 9:
        return 0.0
    sum_areas = sum(cv2.contourArea(c) for c in contours_group)
    if sum_areas == 0:
        return 0.0
    all_pts = np.vstack(contours_group)
    rect = cv2.minAreaRect(all_pts)
    box_area = rect[1][0] * rect[1][1]
    return float(sum_areas / box_area) if box_area > 0 else 0.0


def select_nine_stickers_detailed(candidates,
                                   candidate_limit=DEFAULT_CANDIDATE_LIMIT,
                                   compactness_threshold=DEFAULT_COMPACTNESS_THRESHOLD,
                                   spread_mult=DEFAULT_SPREAD_TOLERANCE_MULT):
    """
    Exposes fast-path candidate group, compactness scores, and fallback details.
    Returns a dict with:
      - 'top_candidates': top N candidates by area
      - 'fast_path_group': 9 contours picked by min area spread
      - 'fast_path_score': float compactness score
      - 'fast_path_passed': bool
      - 'fallback_triggered': bool
      - 'fallback_group': 9 contours chosen by exhaustive search (or None)
      - 'fallback_score': float score (or None)
      - 'fallback_combos_evaluated': int
      - 'final_group': chosen 9 contours (or None)
      - 'final_score': float score
    """
    res = {
        'top_candidates': [],
        'fast_path_group': None,
        'fast_path_score': 0.0,
        'fast_path_passed': False,
        'fallback_triggered': False,
        'fallback_group': None,
        'fallback_score': 0.0,
        'fallback_combos_evaluated': 0,
        'final_group': None,
        'final_score': 0.0,
    }

    if len(candidates) < 9:
        return res

    by_area = sorted(candidates, key=cv2.contourArea, reverse=True)
    top_candidates = by_area[:candidate_limit]
    res['top_candidates'] = top_candidates

    # Sort ascending for spread scan
    subset = sorted(top_candidates, key=cv2.contourArea)

    min_spread, best_i = float('inf'), -1
    for i in range(len(subset) - 8):
        spread = cv2.contourArea(subset[i + 8]) - cv2.contourArea(subset[i])
        if spread < min_spread:
            min_spread, best_i = spread, i
    fast_path_group = subset[best_i:best_i + 9]
    fast_score = calculate_compactness_score(fast_path_group)

    res['fast_path_group'] = fast_path_group
    res['fast_path_score'] = fast_score

    if fast_score >= compactness_threshold:
        res['fast_path_passed'] = True
        res['final_group'] = fast_path_group
        res['final_score'] = fast_score
        return res

    # Fallback path triggered
    res['fast_path_passed'] = False
    res['fallback_triggered'] = True
    spread_tolerance = min_spread * spread_mult
    best_score, best_group = fast_score, fast_path_group
    combos_evaluated = 0

    for combo in itertools.combinations(subset, 9):
        combos_evaluated += 1
        areas = [cv2.contourArea(c) for c in combo]
        if (max(areas) - min(areas)) > spread_tolerance:
            continue
        score = calculate_compactness_score(combo)
        if score > best_score:
            best_score, best_group = score, list(combo)

    res['fallback_group'] = best_group
    res['fallback_score'] = best_score
    res['fallback_combos_evaluated'] = combos_evaluated
    res['final_group'] = best_group
    res['final_score'] = best_score
    return res


def select_nine_stickers(candidates,
                         candidate_limit=DEFAULT_CANDIDATE_LIMIT,
                         compactness_threshold=DEFAULT_COMPACTNESS_THRESHOLD,
                         spread_mult=DEFAULT_SPREAD_TOLERANCE_MULT):
    """Backwards-compatible wrapper returning only the 9 chosen contours."""
    info = select_nine_stickers_detailed(candidates, candidate_limit, compactness_threshold, spread_mult)
    return info['final_group']


# ----------------------------------------------------------------------------
# Stage 4 — Perspective normalization
# ----------------------------------------------------------------------------
def order_points(pts):
    """[top-left, top-right, bottom-right, bottom-left]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_warp_matrix(nine_contours, dst_size=DST_SIZE):
    """Calculate the perspective transform matrix M for the 9 stickers."""
    all_pts = np.vstack(nine_contours)
    rect = cv2.minAreaRect(all_pts)
    box = cv2.boxPoints(rect)
    src = order_points(box)
    dst = np.float32([[0, 0], [dst_size - 1, 0],
                       [dst_size - 1, dst_size - 1], [0, dst_size - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, box


def warp_face(img, nine_contours, dst_size=DST_SIZE):
    M, _ = get_warp_matrix(nine_contours, dst_size)
    return cv2.warpPerspective(img, M, (dst_size, dst_size))


# ----------------------------------------------------------------------------
# Stage 5 — Ordering + average colors
# ----------------------------------------------------------------------------
def order_nine_by_grid(nine_contours):
    """Order 9 contours row-major: top-left (0) to bottom-right (8)."""
    items = []
    for c in nine_contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            cx, cy, _, _ = cv2.boundingRect(c)
        else:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        items.append((cy, cx, c))
    items.sort(key=lambda t: t[0])
    rows = [items[i:i + 3] for i in range(0, 9, 3)]
    ordered = []
    for row in rows:
        row.sort(key=lambda t: t[1])
        ordered.extend(c for (_, _, c) in row)
    return ordered


def average_sticker_colors(warped_img, ordered_contours):
    """Compute average BGR color for each ordered sticker contour."""
    colors = []
    for c in ordered_contours:
        mask = np.zeros(warped_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_bgr = cv2.mean(warped_img, mask=mask)[:3]
        colors.append(mean_bgr)
    return colors  # 9 BGR tuples, row-major (index 4 = center)


def process_face_detailed(img_or_path,
                         saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
                         lower_v=DEFAULT_GLOBAL_HSV_LOWER_V,
                         white_v=DEFAULT_WHITE_VALUE_THRESHOLD,
                         min_area=DEFAULT_MIN_CONTOUR_AREA,
                         ratio_threshold=DEFAULT_RATIO_THRESHOLD,
                         candidate_limit=DEFAULT_CANDIDATE_LIMIT,
                         compactness_threshold=DEFAULT_COMPACTNESS_THRESHOLD,
                         spread_mult=DEFAULT_SPREAD_TOLERANCE_MULT):
    """
    Run single-face pipeline and return all intermediate data for visualization.
    """
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    if img is None:
        return None

    color_mask, white_mask, cleaned_mask = get_masks(img, saturation_threshold, lower_v, white_v)
    raw_contours = find_all_contours(cleaned_mask)
    filtered_candidates = filter_rectangular_candidates(raw_contours, min_area, ratio_threshold)
    selection_info = select_nine_stickers_detailed(
        filtered_candidates, candidate_limit, compactness_threshold, spread_mult
    )

    result = {
        'img': img,
        'color_mask': color_mask,
        'white_mask': white_mask,
        'cleaned_mask': cleaned_mask,
        'raw_contours': raw_contours,
        'filtered_candidates': filtered_candidates,
        'selection_info': selection_info,
        'warped': None,
        'ordered_contours': None,
        'colors': None,
    }

    nine = selection_info['final_group']
    if nine is None or len(nine) != 9:
        return result

    # Warp image to 300x300
    M, box = get_warp_matrix(nine, DST_SIZE)
    warped = cv2.warpPerspective(img, M, (DST_SIZE, DST_SIZE))
    result['warped'] = warped

    # Try re-segmenting on warped image
    warped_mask = segment_stickers(warped, saturation_threshold, lower_v, white_v)
    warped_raw = find_all_contours(warped_mask)
    warped_filtered = filter_rectangular_candidates(warped_raw, min_area=30, ratio_threshold=0.4)
    warped_sel = select_nine_stickers_detailed(warped_filtered, candidate_limit=15,
                                               compactness_threshold=0.5, spread_mult=2.5)

    if warped_sel['final_group'] is not None and len(warped_sel['final_group']) == 9:
        ordered = order_nine_by_grid(warped_sel['final_group'])
    else:
        # Fallback: project original 9 contours through warp matrix M
        projected_nine = []
        for c in nine:
            c_f32 = c.astype(np.float32)
            c_proj = cv2.perspectiveTransform(c_f32, M)
            projected_nine.append(np.int32(c_proj))
        ordered = order_nine_by_grid(projected_nine)

    result['ordered_contours'] = ordered
    result['colors'] = average_sticker_colors(warped, ordered)
    return result


def process_face(image_path,
                 lower_v=DEFAULT_GLOBAL_HSV_LOWER_V,
                 white_v=DEFAULT_WHITE_VALUE_THRESHOLD):
    """Original signature convenience method."""
    details = process_face_detailed(image_path, lower_v=lower_v, white_v=white_v)
    if details is None or details['colors'] is None:
        return None
    return details['colors']


# ----------------------------------------------------------------------------
# Stage 6 — Color classification
# ----------------------------------------------------------------------------
def bgr_to_hsv(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]


def bgr_to_lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)


def circular_hue_distance(h1, h2):
    diff = abs(int(h1) - int(h2))
    return min(diff, 180 - diff)


def classify_sticker(bgr, center_map):
    """Classify a sticker BGR into one of the 6 center faces using CIE Lab perceptual distance."""
    lab = bgr_to_lab(bgr)
    best_face, best_dist = None, float('inf')
    for face, center_bgr in center_map.items():
        dist = np.linalg.norm(lab - bgr_to_lab(center_bgr))
        if dist < best_dist:
            best_dist, best_face = dist, face
    return best_face


# ----------------------------------------------------------------------------
# Stage 7 — Validation and one-color correction
# ----------------------------------------------------------------------------
U, R, F, D, L, B = 0, 9, 18, 27, 36, 45

STICKER_MAP_CORNERS = [
    (U + 6, L + 2, F + 0), (U + 8, F + 2, R + 0),
    (U + 2, R + 2, B + 0), (U + 0, B + 2, L + 0),
    (D + 0, F + 6, L + 8), (D + 2, R + 6, F + 8),
    (D + 8, B + 6, R + 8), (D + 6, L + 6, B + 8),
]

STICKER_MAP_EDGES = [
    (U + 7, F + 1), (U + 5, R + 1), (U + 1, B + 1), (U + 3, L + 1),
    (F + 3, L + 5), (F + 5, R + 3), (B + 3, R + 5), (B + 5, L + 3),
    (D + 1, F + 7), (D + 5, R + 7), (D + 7, B + 7), (D + 3, L + 7),
]


def build_master_inventory(u='U', r='R', f='F', d='D', l='L', b='B'):
    legal_corners = [
        frozenset({u, l, f}), frozenset({u, f, r}),
        frozenset({u, r, b}), frozenset({u, b, l}),
        frozenset({d, f, l}), frozenset({d, r, f}),
        frozenset({d, b, r}), frozenset({d, l, b}),
    ]
    legal_edges = [
        frozenset({u, f}), frozenset({u, r}), frozenset({u, b}), frozenset({u, l}),
        frozenset({f, l}), frozenset({f, r}), frozenset({b, r}), frozenset({b, l}),
        frozenset({d, f}), frozenset({d, r}), frozenset({d, b}), frozenset({d, l}),
    ]
    return legal_corners, legal_edges


def validate_and_correct_detailed(facelets, bgr_by_index, center_map_by_letter):
    """
    Validates corners/edges and applies one-color repair.
    Returns:
      (corrected_facelets, invalid_corners, missing_corners,
       invalid_edges, missing_edges, repairs_dict)
    """
    legal_corners, legal_edges = build_master_inventory()

    def check_pieces(sticker_groups, legal_set):
        invalid, seen_ok = [], []
        for group in sticker_groups:
            colors = [facelets[i] for i in group]
            if len(set(colors)) != len(colors):
                invalid.append(group)
                continue
            fs = frozenset(colors)
            if fs not in legal_set:
                invalid.append(group)
            else:
                seen_ok.append(fs)
        missing = [c for c in legal_set if c not in seen_ok]
        return invalid, missing

    invalid_corners, missing_corners = check_pieces(STICKER_MAP_CORNERS, legal_corners)
    invalid_edges, missing_edges = check_pieces(STICKER_MAP_EDGES, legal_edges)

    corrected = list(facelets)
    repairs = {}

    def repair(invalid_groups, missing_sets):
        if len(invalid_groups) != len(missing_sets):
            return  # ambiguous mismatch count — leave uncorrected
        for group, missing_piece in zip(invalid_groups, missing_sets):
            current = frozenset(corrected[i] for i in group)
            need = set(missing_piece) - current
            if len(need) != 1:
                continue
            missing_color = need.pop()
            missing_center_lab = bgr_to_lab(center_map_by_letter[missing_color])

            best_idx, best_dist = None, float('inf')
            for idx in group:
                dist = np.linalg.norm(bgr_to_lab(bgr_by_index[idx]) - missing_center_lab)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            repairs[best_idx] = (corrected[best_idx], missing_color)
            corrected[best_idx] = missing_color

    repair(invalid_corners, missing_corners)
    repair(invalid_edges, missing_edges)

    return corrected, invalid_corners, missing_corners, invalid_edges, missing_edges, repairs


def validate_and_correct(facelets, bgr_by_index, center_map_by_letter):
    corrected, _, _, _, _, _ = validate_and_correct_detailed(facelets, bgr_by_index, center_map_by_letter)
    return corrected


# ----------------------------------------------------------------------------
# Stage 8 — Solve
# ----------------------------------------------------------------------------
def solve_cube(facelet_string):
    return kociemba.solve(facelet_string)
