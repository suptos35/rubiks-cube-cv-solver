"""
Rubik's Cube Pipeline — Interactive Native OpenCV Debug Visualizer.
Features:
- Single-window OpenCV GUI with live-updating trackbars
- Grid Overview (all 10 stages) + Click-to-Zoom High-Resolution Inspector
- Full keyboard and mouse traversal across images and pipeline stages
- Interactive Tooltip & Status Bar explaining what every stage and slider does
- Mode 0 (Single Face Debug) and Mode 1 (All Six Faces Validation + Solver)
- On-screen slider documentation & debugging guide overlay (toggle with 'H')
"""

import os
import sys
import glob
import cv2
import numpy as np
import pipeline

WINDOW_NAME = "Rubik's Pipeline Debug Visualizer"
SAMPLE_DIR = "sample_faces"

# Color Palette (BGR)
COLOR_BG = (24, 24, 28)
COLOR_PANEL_BG = (32, 32, 38)
COLOR_PANEL_HEADER = (42, 42, 50)
COLOR_BORDER = (55, 55, 65)
COLOR_HOVER_BORDER = (0, 220, 255)
COLOR_ACTIVE_BORDER = (0, 255, 120)
COLOR_TEXT_BRIGHT = (245, 245, 245)
COLOR_TEXT_MAIN = (210, 210, 215)
COLOR_TEXT_MUTED = (150, 150, 160)
COLOR_ACCENT = (0, 180, 255)

# Overlay Colors for Contours / Pipeline Stages
COLOR_GRAY = (140, 140, 140)       # Stage 3: Raw contours
COLOR_YELLOW = (0, 235, 255)       # Stage 4: Filtered survivors
COLOR_ORANGE = (0, 140, 255)       # Stage 5: Top-N candidates
COLOR_GREEN = (0, 240, 60)         # Stage 6/8: Final 9 stickers
COLOR_CYAN = (255, 220, 0)         # Fast-path candidate group (when failed)
COLOR_RED = (50, 50, 255)          # Invalid/error
COLOR_BUTTON_BG = (48, 48, 58)
COLOR_BUTTON_BORDER = (75, 75, 90)

CANVAS_W = 1440
CANVAS_H = 880
TOP_BAR_H = 46
FOOTER_H = 68

# Grid Dimensions (5 cols x 2 rows)
GRID_COLS = 5
GRID_ROWS = 2
PANEL_W = (CANVAS_W - 20) // GRID_COLS
PANEL_H = (CANVAS_H - TOP_BAR_H - FOOTER_H - 15) // GRID_ROWS
HEADER_H = 30

# Stage Information & Debug Guidance
STAGE_DOCS = [
    {
        "num": 1,
        "name": "Original Photo",
        "desc": "Untouched source photo loaded from sample_faces/ folder.",
        "debug": "Inspect image lighting, glare reflections, tilt angle, and contrast against background."
    },
    {
        "num": 2,
        "name": "Masks (Col/Wht/Clean)",
        "desc": "Color mask (S>=Sat, V>=LowerV), White mask (S<Sat, V>=WhiteV), and Cleaned mask (Open+Close).",
        "debug": "If background merges with stickers: RAISE WhiteV. If dark stickers missed: LOWER LowerV/Saturation."
    },
    {
        "num": 3,
        "name": "All Raw Contours",
        "desc": "Every external contour detected on cleaned binary mask before any geometric filtering.",
        "debug": "If count > 100: masks are too noisy (adjust WhiteV/LowerV). If count < 9: segmentation failed."
    },
    {
        "num": 4,
        "name": "Area + Ratio Filter",
        "desc": "Contours passing MinArea and rectangularity (Contour Area / minAreaRect Area >= RatioX100).",
        "debug": "If a real sticker was dropped: LOWER RatioX100 (e.g. 50) or LOWER MinArea (e.g. 50-80)."
    },
    {
        "num": 5,
        "name": "Top-N Candidates",
        "desc": "The N largest surviving contours sorted by area (bounded by CandidateLimit for speed).",
        "debug": "If a real sticker is excluded: background noise took its slot. RAISE CandidateLimit (e.g. 20-25)."
    },
    {
        "num": 6,
        "name": "Fast-Path 9 Selection",
        "desc": "Sliding window of 9 with smallest area spread. Accepted if Compactness >= CompactnessX100.",
        "debug": "Green = Passed. Cyan = Failed. If score is ~0.65-0.69 and stickers are correct, lower CompactnessX100."
    },
    {
        "num": 7,
        "name": "Combinatorial Fallback",
        "desc": "Exhaustive combinations within 1.5x area spread. Evaluates compactness across candidates.",
        "debug": "Triggers only when fast-path fails compactness. Shows number of combinations evaluated."
    },
    {
        "num": 8,
        "name": "Final Selected 9",
        "desc": "The 9 final accepted sticker regions, labeled 1 to 9 with final compactness score.",
        "debug": "Verify all 9 correspond to the physical face stickers before proceeding to color extraction."
    },
    {
        "num": 9,
        "name": "Warped Face (300x300)",
        "desc": "Perspective-warped normalized square with ordered sticker boundaries and grid coordinates.",
        "debug": "Check whether all 9 stickers align cleanly to a 3x3 grid without heavy perspective distortion."
    },
    {
        "num": 10,
        "name": "Sticker BGR Colors",
        "desc": "3x3 grid of sticker color swatches with mean BGR values. Center sticker (#5) is face reference.",
        "debug": "Inspect measured BGR values. Center swatch determines this face's reference color."
    }
]

SLIDER_DOCS = [
    ("Image", "0 to N-1", "Selects active photo from sample_faces/ folder."),
    ("Saturation", "0 to 255 (def: 40)", "Separates chromatic colors (S >= Sat) from white/gray (S < Sat)."),
    ("LowerV", "0 to 255 (def: 80)", "Brightness floor for colored pixels. Filters out shadows and black plastic."),
    ("WhiteV", "0 to 255 (def: 200)", "Brightness floor for white stickers. Higher = avoids white paper background leak."),
    ("MinArea", "0 to 1500 (def: 100)", "Minimum contour pixel area. Rejects dust, glare flecks, and small noise."),
    ("RatioX100", "0 to 100 (def: 60)", "Rectangularity threshold (area / box_area). Rubik's stickers are squarish."),
    ("CandidateLimit", "9 to 35 (def: 15)", "Top candidates kept for selection. Bounds combination complexity C(N, 9)."),
    ("CompactnessX100", "0 to 100 (def: 70)", "Area sum / bounding box. Fast-path accepts immediately if score >= threshold."),
    ("Mode", "0 or 1", "0 = Single-Face Stage Debug; 1 = All 6 Faces Summary, Center Map & Solver.")
]


def get_image_files(sample_dir=SAMPLE_DIR):
    """Retrieve and sort image files from sample_dir."""
    if not os.path.exists(sample_dir):
        return []
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(sample_dir, ext)))
    return sorted(files)


def fit_image(img, target_w, target_h, bg_color=COLOR_PANEL_BG):
    """Fit an image inside target_w x target_h while preserving aspect ratio."""
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    if img is None:
        return canvas

    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    ih, iw = img_bgr.shape[:2]
    if iw == 0 or ih == 0:
        return canvas

    scale = min(target_w / iw, target_h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def draw_contours_labeled(base_img, contours, color, thickness=2, label_start=1, font_scale=0.45):
    """Draw contours with numbered centroid labels."""
    canvas = base_img.copy()
    cv2.drawContours(canvas, contours, -1, color, thickness, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
        label = str(label_start + i)
        cv2.putText(canvas, label, (cx - 7, cy + 5), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx - 7, cy + 5), font, font_scale, color, 1, cv2.LINE_AA)
    return canvas


def render_masks_panel(data, target_w, target_h):
    """Render 3 masks (color, white, cleaned) stacked cleanly with headers."""
    canvas = np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
    masks = [data['color_mask'], data['white_mask'], data['cleaned_mask']]
    labels = ["Color Mask (S>=Sat, V>=LowerV)", "White Mask (S<Sat, V>=WhiteV)", "Cleaned Mask (Open+Close)"]
    sub_h = target_h // 3

    for i, (m, lbl) in enumerate(zip(masks, labels)):
        sy = i * sub_h
        cv2.rectangle(canvas, (4, sy + 2), (target_w - 4, sy + 20), (45, 45, 55), -1)
        cv2.putText(canvas, lbl, (10, sy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
        m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        fit_m = fit_image(m_bgr, target_w - 12, sub_h - 26, bg_color=COLOR_PANEL_BG)
        canvas[sy + 22:sy + 22 + (sub_h - 26), 6:6 + (target_w - 12)] = fit_m
        if i < 2:
            cv2.line(canvas, (0, sy + sub_h - 1), (target_w, sy + sub_h - 1), COLOR_BORDER, 1)
    return canvas


def render_swatches_panel(colors, target_w, target_h):
    """Render 3x3 grid of sticker color swatches with BGR readouts."""
    canvas = np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
    if not colors or len(colors) != 9:
        cv2.putText(canvas, "No sticker colors extracted", (20, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        return canvas

    pad = 10
    grid_w = target_w - pad * 2
    grid_h = target_h - pad * 2
    cell_w = grid_w // 3
    cell_h = grid_h // 3

    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            b, g, r = [int(v) for v in colors[idx]]
            x = pad + col * cell_w
            y = pad + row * cell_h

            # Swatch box
            cv2.rectangle(canvas, (x + 3, y + 3), (x + cell_w - 3, y + cell_h - 24), (b, g, r), -1)
            cv2.rectangle(canvas, (x + 3, y + 3), (x + cell_w - 3, y + cell_h - 24), (65, 65, 75), 1)

            # Label
            tag = f"#{idx + 1}" + (" (Center)" if idx == 4 else "")
            cv2.putText(canvas, tag, (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, tag, (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            bgr_txt = f"{b},{g},{r}"
            cv2.putText(canvas, bgr_txt, (x + 5, y + cell_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    return canvas


def get_stage_rendered_image(stage_idx, data, params, target_w, target_h):
    """Render the graphic content for a specific pipeline stage."""
    img = data['img']
    raw_contours = data['raw_contours']
    filtered = data['filtered_candidates']
    sel = data['selection_info']
    warped = data['warped']
    colors = data['colors']

    if stage_idx == 0:  # 1. Original
        return fit_image(img, target_w, target_h)
    elif stage_idx == 1:  # 2. Masks
        return render_masks_panel(data, target_w, target_h)
    elif stage_idx == 2:  # 3. All Raw
        img_raw = draw_contours_labeled(img, raw_contours, COLOR_GRAY, thickness=1)
        return fit_image(img_raw, target_w, target_h)
    elif stage_idx == 3:  # 4. Filtered
        img_filt = draw_contours_labeled(img, filtered, COLOR_YELLOW, thickness=2)
        return fit_image(img_filt, target_w, target_h)
    elif stage_idx == 4:  # 5. Top-N
        top_n = sel['top_candidates']
        img_top = draw_contours_labeled(img, top_n, COLOR_ORANGE, thickness=3)
        return fit_image(img_top, target_w, target_h)
    elif stage_idx == 5:  # 6. Fast-path
        fp_group = sel['fast_path_group']
        col = COLOR_GREEN if sel['fast_path_passed'] else COLOR_CYAN
        if fp_group:
            img_fp = draw_contours_labeled(img, fp_group, col, thickness=3)
            return fit_image(img_fp, target_w, target_h)
        else:
            canvas = np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
            cv2.putText(canvas, "Not enough candidates (<9)", (20, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1, cv2.LINE_AA)
            return canvas
    elif stage_idx == 6:  # 7. Fallback
        canvas = np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
        if sel['fallback_triggered'] and sel['fallback_group']:
            img_fb = draw_contours_labeled(img, sel['fallback_group'], COLOR_ORANGE, thickness=3)
            fit_fb = fit_image(img_fb, target_w, target_h - 30)
            canvas[:target_h - 30, :] = fit_fb
            txt = f"Fallback score: {sel['fallback_score']:.2f} (tried {sel['fallback_combos_evaluated']})"
            cv2.putText(canvas, txt, (10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_YELLOW, 1, cv2.LINE_AA)
            return canvas
        else:
            status = "Fast-path accepted (Fallback not needed)" if sel['fast_path_passed'] else "No fallback candidates (<9)"
            col = COLOR_GREEN if sel['fast_path_passed'] else COLOR_TEXT_MUTED
            cv2.putText(canvas, status, (15, target_h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Fast score: {sel['fast_path_score']:.2f} >= {params['compact_th']:.2f}",
                        (15, target_h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            return canvas
    elif stage_idx == 7:  # 8. Final 9
        if sel['final_group']:
            img_final = draw_contours_labeled(img, sel['final_group'], COLOR_GREEN, thickness=3)
            return fit_image(img_final, target_w, target_h)
        else:
            canvas = np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
            cv2.putText(canvas, "Selection failed (<9 stickers)", (15, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1, cv2.LINE_AA)
            return canvas
    elif stage_idx == 8:  # 9. Warped Face
        if warped is not None:
            warped_disp = warped.copy()
            if data['ordered_contours']:
                cv2.drawContours(warped_disp, data['ordered_contours'], -1, COLOR_GREEN, 2, cv2.LINE_AA)
                for i, c in enumerate(data['ordered_contours']):
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        cv2.putText(warped_disp, str(i + 1), (cx - 6, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(warped_disp, str(i + 1), (cx - 6, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return fit_image(warped_disp, target_w, target_h)
        else:
            canvas = np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
            cv2.putText(canvas, "Warping unavailable", (20, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            return canvas
    elif stage_idx == 9:  # 10. Swatches
        return render_swatches_panel(colors, target_w, target_h)

    return np.full((target_h, target_w, 3), COLOR_PANEL_BG, dtype=np.uint8)


def get_stage_subtitles(stage_idx, data, params):
    """Return compact subtitle metric for panel header."""
    sel = data['selection_info']
    if stage_idx == 0:
        return f"{data['img'].shape[1]}x{data['img'].shape[0]}"
    elif stage_idx == 1:
        return "Col/Wht/Clean"
    elif stage_idx == 2:
        return f"Count: {len(data['raw_contours'])}"
    elif stage_idx == 3:
        return f"Count: {len(data['filtered_candidates'])}"
    elif stage_idx == 4:
        return f"Top {len(sel['top_candidates'])}"
    elif stage_idx == 5:
        st = "PASS" if sel['fast_path_passed'] else "FAIL"
        return f"Score: {sel['fast_path_score']:.2f} [{st}]"
    elif stage_idx == 6:
        return f"{sel['fallback_combos_evaluated']} combos" if sel['fallback_triggered'] else "Not triggered"
    elif stage_idx == 7:
        return f"Score: {sel['final_score']:.2f}"
    elif stage_idx == 8:
        return "300x300 Norm"
    elif stage_idx == 9:
        return "3x3 Row-Major"
    return ""


def render_top_bar(canvas, state, image_files):
    """Draw top header with interactive button guides and current status."""
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, TOP_BAR_H), COLOR_PANEL_HEADER, -1)
    cv2.line(canvas, (0, TOP_BAR_H), (CANVAS_W, TOP_BAR_H), COLOR_BORDER, 1)

    cur_idx = state['img_idx']
    total_imgs = len(image_files)
    cur_file = os.path.basename(image_files[cur_idx]) if image_files else "None"

    # Navigation buttons
    btn_y = 8
    btn_h = 28

    # Button: Prev [A]
    cv2.rectangle(canvas, (10, btn_y), (85, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (10, btn_y), (85, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, "< Prev (A)", (16, btn_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # Photo info
    img_tag = f"Photo {cur_idx + 1}/{total_imgs}: {cur_file}"
    cv2.putText(canvas, img_tag, (95, btn_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)

    # Button: Next [D]
    nx_btn_x = 95 + cv2.getTextSize(img_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 12
    cv2.rectangle(canvas, (nx_btn_x, btn_y), (nx_btn_x + 75, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (nx_btn_x, btn_y), (nx_btn_x + 75, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, "Next (D) >", (nx_btn_x + 6, btn_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # Mode Indicator / Toggle Button
    mode_x = CANVAS_W - 470
    mode_text = "Mode: [0] Single Face Debug" if state['mode'] == 0 else "Mode: [1] All Six Faces Summary"
    mode_col = COLOR_ACCENT if state['mode'] == 0 else COLOR_YELLOW
    cv2.rectangle(canvas, (mode_x, btn_y), (mode_x + 230, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (mode_x, btn_y), (mode_x + 230, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, mode_text, (mode_x + 8, btn_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_col, 1, cv2.LINE_AA)

    # View Mode Button (Grid vs Zoom)
    view_x = CANVAS_W - 230
    view_text = "Zoom: On [Space]" if state['view_mode'] == 'zoom' else "Zoom: Off (Click to zoom)"
    view_col = COLOR_ACTIVE_BORDER if state['view_mode'] == 'zoom' else COLOR_TEXT_MUTED
    cv2.rectangle(canvas, (view_x, btn_y), (view_x + 155, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (view_x, btn_y), (view_x + 155, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, view_text, (view_x + 8, btn_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, view_col, 1, cv2.LINE_AA)

    # Help toggle [H]
    help_x = CANVAS_W - 65
    cv2.rectangle(canvas, (help_x, btn_y), (help_x + 55, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (help_x, btn_y), (help_x + 55, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    h_col = COLOR_YELLOW if state['show_help'] else COLOR_TEXT_MUTED
    cv2.putText(canvas, "Help (H)", (help_x + 5, btn_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, h_col, 1, cv2.LINE_AA)


def render_footer_bar(canvas, hovered_stage, state):
    """Draw footer info explaining hovered stage or active controls."""
    fy = CANVAS_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (CANVAS_W, CANVAS_H), (20, 20, 24), -1)
    cv2.line(canvas, (0, fy), (CANVAS_W, fy), COLOR_BORDER, 1)

    if state['mode'] == 0:
        if hovered_stage is not None and 0 <= hovered_stage < len(STAGE_DOCS):
            doc = STAGE_DOCS[hovered_stage]
            title = f"Stage {doc['num']}: {doc['name']}  |  {doc['desc']}"
            cv2.putText(canvas, title, (14, fy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA)

            debug_tip = f"Debugging Guide: {doc['debug']}"
            cv2.putText(canvas, debug_tip, (14, fy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

            action_tip = "[Click panel or press 1-9/0 to Zoom in/out]  [A/D or Left/Right: change photo]  [M: toggle Mode]"
            cv2.putText(canvas, action_tip, (CANVAS_W - 640, fy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Hover over any panel for stage details and debugging guidance. Click any panel to view full-size.",
                        (14, fy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
            shortcuts = "Shortcuts: [A/D] Prev/Next Photo | [1-9, 0] Zoom Stage | [Space] Toggle Zoom | [M] Toggle Mode | [H] Slider Guide | [Q/Esc] Quit"
            cv2.putText(canvas, shortcuts, (14, fy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Mode 1: All-Six-Faces Summary. Assembles centers, classifies 54 stickers, validates topology, and solves via Kociemba.",
                    (14, fy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[M: return to Single-Face Debug]  [A/D: change reference image]  [H: show Slider Parameter Guide]",
                    (14, fy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)


def render_grid_view(canvas, data, params, state):
    """Render all 10 stages in a 5x2 grid with interactive hover borders."""
    grid_y_start = TOP_BAR_H + 8
    pad_x = 4
    pad_y = 4

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            stage_idx = row * GRID_COLS + col
            x = 8 + col * (PANEL_W + pad_x)
            y = grid_y_start + row * (PANEL_H + pad_y)

            is_hovered = (state['hovered_stage'] == stage_idx)
            border_col = COLOR_HOVER_BORDER if is_hovered else COLOR_BORDER

            # Panel background & border
            cv2.rectangle(canvas, (x, y), (x + PANEL_W, y + PANEL_H), COLOR_PANEL_BG, -1)
            cv2.rectangle(canvas, (x, y), (x + PANEL_W, y + PANEL_H), border_col, 2 if is_hovered else 1)

            # Header bar
            cv2.rectangle(canvas, (x, y), (x + PANEL_W, y + HEADER_H), COLOR_PANEL_HEADER, -1)
            cv2.line(canvas, (x, y + HEADER_H), (x + PANEL_W, y + HEADER_H), border_col, 1)

            doc = STAGE_DOCS[stage_idx]
            title = f"{doc['num']}. {doc['name']}"
            subtitle = get_stage_subtitles(stage_idx, data, params)

            cv2.putText(canvas, title, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
            if subtitle:
                sub_w = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
                cv2.putText(canvas, subtitle, (x + PANEL_W - sub_w - 8, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

            # Stage image content
            content_w = PANEL_W - 8
            content_h = PANEL_H - HEADER_H - 8
            stage_img = get_stage_rendered_image(stage_idx, data, params, content_w, content_h)
            canvas[y + HEADER_H + 4:y + HEADER_H + 4 + content_h, x + 4:x + 4 + content_w] = stage_img


def render_zoomed_view(canvas, data, params, state):
    """Render high-resolution focused stage inspector."""
    stage_idx = state['focused_stage']
    doc = STAGE_DOCS[stage_idx]

    main_y = TOP_BAR_H + 8
    main_h = CANVAS_H - TOP_BAR_H - FOOTER_H - 16

    # Left: High-resolution image canvas (width: 940)
    img_w = 940
    cv2.rectangle(canvas, (10, main_y), (10 + img_w, main_y + main_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (10, main_y), (10 + img_w, main_y + main_h), COLOR_BORDER, 1)

    # Render image at high resolution
    highres_img = get_stage_rendered_image(stage_idx, data, params, img_w - 16, main_h - 16)
    canvas[main_y + 8:main_y + 8 + (main_h - 16), 18:18 + (img_w - 16)] = highres_img

    # Right: Stage Inspector Sidebar (width: 460)
    side_x = 10 + img_w + 14
    side_w = CANVAS_W - side_x - 12
    cv2.rectangle(canvas, (side_x, main_y), (side_x + side_w, main_y + main_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (side_x, main_y), (side_x + side_w, main_y + main_h), COLOR_BORDER, 1)

    # Header
    cv2.rectangle(canvas, (side_x, main_y), (side_x + side_w, main_y + 44), COLOR_PANEL_HEADER, -1)
    cv2.line(canvas, (side_x, main_y + 44), (side_x + side_w, main_y + 44), COLOR_BORDER, 1)
    cv2.putText(canvas, f"Stage {doc['num']}: {doc['name']}", (side_x + 14, main_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_YELLOW, 2, cv2.LINE_AA)

    # Navigation buttons: Prev Stage, Next Stage, Back to Grid
    btn_y = main_y + 56
    btn_h = 32

    # [< Back to Grid]
    cv2.rectangle(canvas, (side_x + 12, btn_y), (side_x + 155, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (side_x + 12, btn_y), (side_x + 155, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, "<< Grid View", (side_x + 22, btn_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # [< Prev Stage (W)]
    cv2.rectangle(canvas, (side_x + 170, btn_y), (side_x + 295, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (side_x + 170, btn_y), (side_x + 295, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, "< Prev (W)", (side_x + 180, btn_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # [Next Stage (S) >]
    cv2.rectangle(canvas, (side_x + 310, btn_y), (side_x + 435, btn_y + btn_h), COLOR_BUTTON_BG, -1)
    cv2.rectangle(canvas, (side_x + 310, btn_y), (side_x + 435, btn_y + btn_h), COLOR_BUTTON_BORDER, 1)
    cv2.putText(canvas, "Next (S) >", (side_x + 322, btn_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)

    # Stage jump numbers: 1 2 3 4 5 6 7 8 9 10
    num_y = btn_y + 44
    cv2.putText(canvas, "Jump to Stage:", (side_x + 14, num_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    box_w = 34
    for i in range(10):
        bx = side_x + 14 + (i % 5) * (box_w + 10)
        by = num_y + 26 + (i // 5) * 32
        is_cur = (i == stage_idx)
        b_col = COLOR_ACTIVE_BORDER if is_cur else COLOR_BUTTON_BG
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + 24), b_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + 24), COLOR_BORDER, 1)
        txt = str(i + 1)
        t_col = (0, 0, 0) if is_cur else COLOR_TEXT_BRIGHT
        cv2.putText(canvas, txt, (bx + (10 if i < 9 else 4), by + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, t_col, 1, cv2.LINE_AA)

    # Details & Metrics
    ty = num_y + 100
    cv2.putText(canvas, "Stage Overview & Purpose:", (side_x + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACCENT, 1, cv2.LINE_AA)
    ty += 24
    # Multi-line description wrap
    desc_words = doc['desc'].split(' ')
    line = ""
    for w in desc_words:
        if len(line + " " + w) < 46:
            line += (" " if line else "") + w
        else:
            cv2.putText(canvas, line, (side_x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
            ty += 20
            line = w
    if line:
        cv2.putText(canvas, line, (side_x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        ty += 28

    # Specific Stage Metrics
    ty += 10
    cv2.putText(canvas, "Current Stage Metrics:", (side_x + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACCENT, 1, cv2.LINE_AA)
    ty += 24
    sel = data['selection_info']
    metrics = [
        f"Raw Contours Detected: {len(data['raw_contours'])}",
        f"Filtered Candidates: {len(data['filtered_candidates'])}",
        f"Fast-Path Score: {sel['fast_path_score']:.3f} (Threshold: {params['compact_th']:.2f})",
        f"Fast-Path Status: {'ACCEPTED (PASS)' if sel['fast_path_passed'] else 'REJECTED (FAIL)'}",
        f"Combinatorial Fallback: {'TRIGGERED' if sel['fallback_triggered'] else 'NOT TRIGGERED'}",
        f"Final Compactness Score: {sel['final_score']:.3f}"
    ]
    for m in metrics:
        col = COLOR_GREEN if "PASS" in m or "ACCEPTED" in m else (COLOR_RED if "FAIL" in m else COLOR_TEXT_MAIN)
        cv2.putText(canvas, "- " + m, (side_x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        ty += 22

    # Debugging Advice
    ty += 14
    cv2.putText(canvas, "How to Debug this Stage:", (side_x + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_YELLOW, 1, cv2.LINE_AA)
    ty += 24
    dbg_words = doc['debug'].split(' ')
    line = ""
    for w in dbg_words:
        if len(line + " " + w) < 46:
            line += (" " if line else "") + w
        else:
            cv2.putText(canvas, line, (side_x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
            ty += 20
            line = w
    if line:
        cv2.putText(canvas, line, (side_x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)


def render_help_overlay(canvas):
    """Render toggleable overlay explaining all sliders and keyboard shortcuts."""
    overlay = canvas.copy()
    box_x = 80
    box_y = 60
    box_w = CANVAS_W - 160
    box_h = CANVAS_H - 140

    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 18, 22), -1)
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_ACCENT, 2)

    # Header
    cv2.putText(overlay, "Rubik's Pipeline — Parameter Sliders & Debugging Cheatsheet",
                (box_x + 20, box_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(overlay, "Press 'H' or Click anywhere to close",
                (box_x + box_w - 280, box_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.line(overlay, (box_x, box_y + 50), (box_x + box_w, box_y + 50), COLOR_BORDER, 1)

    # Sliders table
    sy = box_y + 80
    cv2.putText(overlay, "Slider Name", (box_x + 24, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(overlay, "Range & Default", (box_x + 190, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(overlay, "Function & Debugging Tip", (box_x + 390, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)
    sy += 12
    cv2.line(overlay, (box_x + 20, sy), (box_x + box_w - 20, sy), (60, 60, 70), 1)

    for name, rng, desc in SLIDER_DOCS:
        sy += 28
        cv2.putText(overlay, name, (box_x + 24, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
        cv2.putText(overlay, rng, (box_x + 190, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(overlay, desc, (box_x + 390, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # Keyboard navigation
    sy += 40
    cv2.line(overlay, (box_x + 20, sy), (box_x + box_w - 20, sy), (60, 60, 70), 1)
    sy += 24
    cv2.putText(overlay, "Keyboard Shortcuts:", (box_x + 24, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)
    sy += 24
    keys = [
        "A / D or Left / Right : Step to Previous / Next Photo in sample_faces/",
        "1 to 9, 0 : Instantly zoom into Pipeline Stages 1 through 10",
        "W / S or Up / Down : Step to Previous / Next Stage in Zoom View",
        "Space or Tab : Toggle between 10-Panel Grid View and Zoom View",
        "M : Toggle between Mode 0 (Single-Face Debug) and Mode 1 (All-Six-Faces Summary)",
        "H : Show / Hide this Parameter Sliders & Shortcuts Guide",
        "Q or Esc : Quit the visualizer"
    ]
    for k in keys:
        cv2.putText(overlay, "- " + k, (box_x + 30, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        sy += 22

    # Blend overlay with slight transparency
    cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)


def render_all_six_view(image_files, params, state):
    """Render Mode 1: All six faces center map, 54-sticker net, and solver output."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), COLOR_BG, dtype=np.uint8)

    render_top_bar(canvas, state, image_files)

    if len(image_files) < 6:
        msg = f"Found only {len(image_files)} images in {SAMPLE_DIR}/. Mode 1 requires at least 6 face photos."
        cv2.putText(canvas, msg, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2, cv2.LINE_AA)
        render_footer_bar(canvas, None, state)
        return canvas

    face_order = pipeline.FACE_ORDER
    face_labels = pipeline.FACE_LABELS
    faces_data = {}
    center_map = {}
    all_ok = True

    for idx, f in enumerate(face_order):
        img_path = image_files[idx]
        res = pipeline.process_face_detailed(
            img_path,
            saturation_threshold=params['sat_th'],
            lower_v=params['lower_v'],
            white_v=params['white_v'],
            min_area=params['min_area'],
            ratio_threshold=params['ratio_th'],
            candidate_limit=params['cand_limit'],
            compactness_threshold=params['compact_th']
        )
        if res is None or res['colors'] is None or len(res['colors']) != 9:
            all_ok = False
            faces_data[f] = None
        else:
            faces_data[f] = res
            center_map[face_labels[f]] = res['colors'][4]

    if not all_ok:
        msg = "Failed to detect 9 stickers in one or more faces. Switch to Mode 0 to tune thresholds for failing photos."
        cv2.putText(canvas, msg, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2, cv2.LINE_AA)
        render_footer_bar(canvas, None, state)
        return canvas

    # Classify 54 stickers
    facelets = []
    bgr_by_idx = {}
    sticker_global_idx = 0
    facelet_meta = []

    for f in face_order:
        letter = face_labels[f]
        face_colors = faces_data[f]['colors']
        for local_idx, bgr in enumerate(face_colors):
            classified_letter = pipeline.classify_sticker(bgr, center_map)
            facelets.append(classified_letter)
            bgr_by_idx[sticker_global_idx] = bgr
            facelet_meta.append({
                'face': letter,
                'local_idx': local_idx,
                'global_idx': sticker_global_idx,
                'bgr': bgr,
                'classified': classified_letter,
                'corrected': classified_letter
            })
            sticker_global_idx += 1

    corrected_facelets, inv_c, miss_c, inv_e, miss_e, repairs = pipeline.validate_and_correct_detailed(
        facelets, bgr_by_idx, center_map
    )
    for g_idx, (old_col, new_col) in repairs.items():
        facelet_meta[g_idx]['corrected'] = new_col

    facelet_string = ''.join(corrected_facelets)
    solver_solution = None
    solver_err = None
    try:
        solver_solution = pipeline.solve_cube(facelet_string)
    except Exception as e:
        solver_err = str(e)

    # Left panel: Centers + Topology + Solver
    left_w = 440
    panel_y = TOP_BAR_H + 8
    panel_h = CANVAS_H - TOP_BAR_H - FOOTER_H - 16

    cv2.rectangle(canvas, (12, panel_y), (left_w, panel_y + panel_h), COLOR_PANEL_BG, -1)
    cv2.rectangle(canvas, (12, panel_y), (left_w, panel_y + panel_h), COLOR_BORDER, 1)

    # Reference Centers
    cy = panel_y + 24
    cv2.putText(canvas, "Center Reference Colors (Learned)", (24, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_YELLOW, 1, cv2.LINE_AA)
    cy += 20
    for ltr in ['U', 'R', 'F', 'D', 'L', 'B']:
        cbgr = center_map[ltr]
        b, g, r = [int(v) for v in cbgr]
        cv2.rectangle(canvas, (24, cy), (62, cy + 24), (b, g, r), -1)
        cv2.rectangle(canvas, (24, cy), (62, cy + 24), (90, 90, 100), 1)
        cv2.putText(canvas, f"Face {ltr} : BGR ({b}, {g}, {r})", (74, cy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
        cy += 32

    # Validation
    cy += 10
    cv2.line(canvas, (20, cy), (left_w - 20, cy), (55, 55, 65), 1)
    cy += 22
    cv2.putText(canvas, "Cube Topology Validation", (24, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_YELLOW, 1, cv2.LINE_AA)
    cy += 22
    c_status = f"Corners: {len(inv_c)} invalid, {len(miss_c)} missing"
    cv2.putText(canvas, c_status, (24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COLOR_GREEN if len(inv_c) == 0 else COLOR_RED, 1, cv2.LINE_AA)
    cy += 20
    e_status = f"Edges: {len(inv_e)} invalid, {len(miss_e)} missing"
    cv2.putText(canvas, e_status, (24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COLOR_GREEN if len(inv_e) == 0 else COLOR_RED, 1, cv2.LINE_AA)
    cy += 20
    rep_status = f"One-Color Repairs Applied: {len(repairs)}"
    cv2.putText(canvas, rep_status, (24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COLOR_CYAN if len(repairs) > 0 else COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    if repairs:
        for s_idx, (o_col, n_col) in repairs.items():
            cy += 18
            txt = f"Sticker #{s_idx} ({facelet_meta[s_idx]['face']}): {o_col} -> {n_col}"
            cv2.putText(canvas, txt, (36, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)

    # Solver
    cy += 24
    cv2.line(canvas, (20, cy), (left_w - 20, cy), (55, 55, 65), 1)
    cy += 22
    cv2.putText(canvas, "Kociemba Solver Solution", (24, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_YELLOW, 1, cv2.LINE_AA)
    cy += 22
    cv2.putText(canvas, f"Facelet String: {facelet_string[:27]}...", (24, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cy += 22
    if solver_solution:
        cv2.putText(canvas, "Solution Moves:", (24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GREEN, 1, cv2.LINE_AA)
        cy += 22
        cv2.putText(canvas, solver_solution, (24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, f"Status: {solver_err}", (24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_RED, 1, cv2.LINE_AA)

    # Right: 54-Sticker Cube Net
    net_x_origin = left_w + 50
    net_y_origin = TOP_BAR_H + 20
    face_size = 195
    cell_size = face_size // 3

    face_grid_coords = {
        'U': (net_x_origin + face_size + 15, net_y_origin),
        'L': (net_x_origin, net_y_origin + face_size + 15),
        'F': (net_x_origin + face_size + 15, net_y_origin + face_size + 15),
        'R': (net_x_origin + 2 * (face_size + 15), net_y_origin + face_size + 15),
        'B': (net_x_origin + 3 * (face_size + 15), net_y_origin + face_size + 15),
        'D': (net_x_origin + face_size + 15, net_y_origin + 2 * (face_size + 15)),
    }

    # Legend
    leg_x = net_x_origin + 2 * (face_size + 15)
    leg_y = net_y_origin + 40
    cv2.putText(canvas, "Sticker Labels Legend:", (leg_x, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_BRIGHT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Background: Measured BGR color", (leg_x, leg_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Bold Letter: Classified Facelet", (leg_x, leg_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Cyan: Repaired Sticker (Old -> New)", (leg_x, leg_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)

    face_offsets = {'U': 0, 'R': 9, 'F': 18, 'D': 27, 'L': 36, 'B': 45}

    for ltr, (fx, fy) in face_grid_coords.items():
        cv2.rectangle(canvas, (fx - 3, fy - 3), (fx + face_size + 3, fy + face_size + 3), (60, 60, 70), 1)
        cv2.putText(canvas, f"Face {ltr}", (fx + 4, fy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_YELLOW, 1, cv2.LINE_AA)

        base_idx = face_offsets[ltr]
        for row in range(3):
            for col in range(3):
                local_idx = row * 3 + col
                g_idx = base_idx + local_idx
                meta = facelet_meta[g_idx]
                b, g, r = [int(v) for v in meta['bgr']]

                cx = fx + col * cell_size
                cy = fy + row * cell_size

                cv2.rectangle(canvas, (cx + 2, cy + 2), (cx + cell_size - 2, cy + cell_size - 2), (b, g, r), -1)
                cv2.rectangle(canvas, (cx + 2, cy + 2), (cx + cell_size - 2, cy + cell_size - 2), (50, 50, 50), 1)

                classified = meta['classified']
                corrected = meta['corrected']
                text_col = (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 130 else (255, 255, 255)

                if classified == corrected:
                    cv2.putText(canvas, classified, (cx + cell_size // 2 - 9, cy + cell_size // 2 + 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, text_col, 2, cv2.LINE_AA)
                else:
                    rep_txt = f"{classified}->{corrected}"
                    cv2.rectangle(canvas, (cx + 3, cy + 3), (cx + cell_size - 3, cy + cell_size - 3), (0, 0, 0), -1)
                    cv2.putText(canvas, rep_txt, (cx + 4, cy + cell_size // 2 + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)

    render_footer_bar(canvas, None, state)
    if state['show_help']:
        render_help_overlay(canvas)
    return canvas


def main():
    image_files = get_image_files()
    if not image_files:
        print(f"Error: No images found in '{SAMPLE_DIR}/'. Please add sample photos.")
        sys.exit(1)

    print(f"Loaded {len(image_files)} sample photos from {SAMPLE_DIR}/")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    # Empty dummy callback for OpenCV trackbars
    def nothing(x):
        pass

    # Create Trackbars
    cv2.createTrackbar("Image", WINDOW_NAME, 0, max(0, len(image_files) - 1), nothing)
    cv2.createTrackbar("Saturation", WINDOW_NAME, pipeline.DEFAULT_SATURATION_THRESHOLD, 255, nothing)
    cv2.createTrackbar("LowerV", WINDOW_NAME, pipeline.DEFAULT_GLOBAL_HSV_LOWER_V, 255, nothing)
    cv2.createTrackbar("WhiteV", WINDOW_NAME, pipeline.DEFAULT_WHITE_VALUE_THRESHOLD, 255, nothing)
    cv2.createTrackbar("MinArea", WINDOW_NAME, pipeline.DEFAULT_MIN_CONTOUR_AREA, 1500, nothing)
    cv2.createTrackbar("RatioX100", WINDOW_NAME, int(pipeline.DEFAULT_RATIO_THRESHOLD * 100), 100, nothing)
    cv2.createTrackbar("CandidateLimit", WINDOW_NAME, pipeline.DEFAULT_CANDIDATE_LIMIT, 35, nothing)
    cv2.createTrackbar("CompactnessX100", WINDOW_NAME, int(pipeline.DEFAULT_COMPACTNESS_THRESHOLD * 100), 100, nothing)
    cv2.createTrackbar("Mode", WINDOW_NAME, 0, 1, nothing)

    # App state
    state = {
        'img_idx': 0,
        'mode': 0,
        'view_mode': 'grid',       # 'grid' or 'zoom'
        'focused_stage': 0,        # 0 to 9
        'hovered_stage': None,     # None or 0 to 9
        'show_help': False,
        'mouse_pos': (0, 0),
        'clicked_action': None
    }

    def on_mouse(event, x, y, flags, param):
        state['mouse_pos'] = (x, y)

        # Detect hovered stage in grid view
        if state['mode'] == 0 and state['view_mode'] == 'grid':
            grid_y_start = TOP_BAR_H + 8
            pad_x = 4
            pad_y = 4
            hovered = None
            for row in range(GRID_ROWS):
                for col in range(GRID_COLS):
                    px = 8 + col * (PANEL_W + pad_x)
                    py = grid_y_start + row * (PANEL_H + pad_y)
                    if px <= x <= px + PANEL_W and py <= y <= py + PANEL_H:
                        hovered = row * GRID_COLS + col
                        break
            state['hovered_stage'] = hovered
        else:
            state['hovered_stage'] = None

        if event == cv2.EVENT_LBUTTONDOWN:
            if state['show_help']:
                state['show_help'] = False
                return

            # Top bar button clicks
            if y <= TOP_BAR_H:
                # Prev Button
                if 10 <= x <= 85:
                    if state['img_idx'] > 0:
                        cv2.setTrackbarPos("Image", WINDOW_NAME, state['img_idx'] - 1)
                    return
                # Next Button
                img_tag = f"Photo {state['img_idx'] + 1}/{len(image_files)}: "
                tag_w = cv2.getTextSize(img_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                nx_btn_x = 95 + tag_w + 12
                if nx_btn_x <= x <= nx_btn_x + 75:
                    if state['img_idx'] < len(image_files) - 1:
                        cv2.setTrackbarPos("Image", WINDOW_NAME, state['img_idx'] + 1)
                    return
                # Mode Button
                mode_x = CANVAS_W - 470
                if mode_x <= x <= mode_x + 230:
                    new_mode = 1 if state['mode'] == 0 else 0
                    cv2.setTrackbarPos("Mode", WINDOW_NAME, new_mode)
                    return
                # View Mode Button
                view_x = CANVAS_W - 230
                if view_x <= x <= view_x + 155:
                    state['view_mode'] = 'zoom' if state['view_mode'] == 'grid' else 'grid'
                    return
                # Help Button
                help_x = CANVAS_W - 65
                if help_x <= x <= help_x + 55:
                    state['show_help'] = not state['show_help']
                    return

            # Grid clicks -> zoom into clicked panel
            if state['mode'] == 0:
                if state['view_mode'] == 'grid':
                    if state['hovered_stage'] is not None:
                        state['focused_stage'] = state['hovered_stage']
                        state['view_mode'] = 'zoom'
                elif state['view_mode'] == 'zoom':
                    main_y = TOP_BAR_H + 8
                    img_w = 940
                    side_x = 10 + img_w + 14
                    btn_y = main_y + 56
                    btn_h = 32

                    # [< Back to Grid]
                    if side_x + 12 <= x <= side_x + 155 and btn_y <= y <= btn_y + btn_h:
                        state['view_mode'] = 'grid'
                        return
                    # [< Prev Stage]
                    if side_x + 170 <= x <= side_x + 295 and btn_y <= y <= btn_y + btn_h:
                        state['focused_stage'] = (state['focused_stage'] - 1) % 10
                        return
                    # [Next Stage >]
                    if side_x + 310 <= x <= side_x + 435 and btn_y <= y <= btn_y + btn_h:
                        state['focused_stage'] = (state['focused_stage'] + 1) % 10
                        return
                    # Number buttons 1..10
                    num_y = btn_y + 44
                    box_w = 34
                    for i in range(10):
                        bx = side_x + 14 + (i % 5) * (box_w + 10)
                        by = num_y + 26 + (i // 5) * 32
                        if bx <= x <= bx + box_w and by <= y <= by + 24:
                            state['focused_stage'] = i
                            return

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    cached_data = None
    last_process_params = None

    while True:
        # Read trackbars
        img_idx = cv2.getTrackbarPos("Image", WINDOW_NAME)
        img_idx = min(max(0, img_idx), len(image_files) - 1)
        state['img_idx'] = img_idx

        sat_th = cv2.getTrackbarPos("Saturation", WINDOW_NAME)
        lower_v = cv2.getTrackbarPos("LowerV", WINDOW_NAME)
        white_v = cv2.getTrackbarPos("WhiteV", WINDOW_NAME)
        min_area = cv2.getTrackbarPos("MinArea", WINDOW_NAME)
        ratio_th = max(0.01, cv2.getTrackbarPos("RatioX100", WINDOW_NAME) / 100.0)
        cand_limit = max(9, cv2.getTrackbarPos("CandidateLimit", WINDOW_NAME))
        compact_th = max(0.01, cv2.getTrackbarPos("CompactnessX100", WINDOW_NAME) / 100.0)
        mode = cv2.getTrackbarPos("Mode", WINDOW_NAME)
        state['mode'] = mode

        params = {
            'sat_th': sat_th,
            'lower_v': lower_v,
            'white_v': white_v,
            'min_area': min_area,
            'ratio_th': ratio_th,
            'cand_limit': cand_limit,
            'compact_th': compact_th
        }

        # Check if re-processing required
        current_proc_signature = (img_idx, sat_th, lower_v, white_v, min_area, ratio_th, cand_limit, compact_th)
        if current_proc_signature != last_process_params:
            last_process_params = current_proc_signature
            current_img_path = image_files[img_idx]
            cached_data = pipeline.process_face_detailed(
                current_img_path,
                saturation_threshold=sat_th,
                lower_v=lower_v,
                white_v=white_v,
                min_area=min_area,
                ratio_threshold=ratio_th,
                candidate_limit=cand_limit,
                compactness_threshold=compact_th
            )

        if mode == 0:
            canvas = np.full((CANVAS_H, CANVAS_W, 3), COLOR_BG, dtype=np.uint8)
            render_top_bar(canvas, state, image_files)
            if state['view_mode'] == 'grid':
                render_grid_view(canvas, cached_data, params, state)
            else:
                render_zoomed_view(canvas, cached_data, params, state)
            render_footer_bar(canvas, state['hovered_stage'] if state['view_mode'] == 'grid' else state['focused_stage'], state)
            if state['show_help']:
                render_help_overlay(canvas)
            composite = canvas
        else:
            composite = render_all_six_view(image_files, params, state)

        cv2.imshow(WINDOW_NAME, composite)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or Esc
            if state['show_help']:
                state['show_help'] = False
            elif state['view_mode'] == 'zoom':
                state['view_mode'] = 'grid'
            else:
                break
        elif key == ord('h') or key == ord('H'):
            state['show_help'] = not state['show_help']
        elif key == ord('m') or key == ord('M'):
            new_mode = 1 if mode == 0 else 0
            cv2.setTrackbarPos("Mode", WINDOW_NAME, new_mode)
        elif key == 32 or key == 9:  # Space or Tab: toggle grid / zoom
            state['view_mode'] = 'zoom' if state['view_mode'] == 'grid' else 'grid'
        elif key == ord('a') or key == 81:  # 'a' or Left Arrow: Prev image
            if img_idx > 0:
                cv2.setTrackbarPos("Image", WINDOW_NAME, img_idx - 1)
        elif key == ord('d') or key == 83:  # 'd' or Right Arrow: Next image
            if img_idx < len(image_files) - 1:
                cv2.setTrackbarPos("Image", WINDOW_NAME, img_idx + 1)
        elif key == ord('w') or key == 82:  # 'w' or Up Arrow: Prev stage
            state['focused_stage'] = (state['focused_stage'] - 1) % 10
            state['view_mode'] = 'zoom'
        elif key == ord('s') or key == 84:  # 's' or Down Arrow: Next stage
            state['focused_stage'] = (state['focused_stage'] + 1) % 10
            state['view_mode'] = 'zoom'
        elif ord('1') <= key <= ord('9'):  # Number keys 1-9: Jump directly to stage 1-9
            state['focused_stage'] = key - ord('1')
            state['view_mode'] = 'zoom'
        elif key == ord('0'):  # '0' key: Jump to stage 10
            state['focused_stage'] = 9
            state['view_mode'] = 'zoom'

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
