import os
import sys
import cv2
import numpy as np
from PIL import Image
import pipeline

os.makedirs('assets', exist_ok=True)
sample_path = 'sample_faces/photo_1_2025-10-22_11-28-04.jpg'
res = pipeline.process_face_detailed(sample_path)

TARGET_W, TARGET_H = 700, 700
BANNER_H = 80

def create_card(img_bgr, step_title, step_desc, step_num, total_steps=9):
    h, w = img_bgr.shape[:2]
    avail_h = TARGET_H - BANNER_H
    scale = min(TARGET_W / w, avail_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 33)
    
    y_off = BANNER_H + (avail_h - nh) // 2
    x_off = (TARGET_W - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    
    cv2.rectangle(canvas, (0, 0), (TARGET_W, BANNER_H), (15, 17, 23), -1)
    cv2.line(canvas, (0, BANNER_H), (TARGET_W, BANNER_H), (56, 189, 248), 2)
    
    badge_text = f"STEP {step_num}/{total_steps}"
    cv2.putText(canvas, badge_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, step_title, (160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, step_desc, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 190, 205), 1, cv2.LINE_AA)
    
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_contours(img, contours, color=(0, 255, 0), thickness=2, label=False):
    out = img.copy()
    cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    if label:
        for i, c in enumerate(contours):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
            lbl = str(i + 1)
            cv2.putText(out, lbl, (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, lbl, (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out

frames = []
durations = []

# Step 1: Original Image
orig = res['img']
frames.append(create_card(orig, "Original Camera Input", "Raw photo captured from user camera or phone", 1))
durations.append(1500)

# Step 2: Chromatic Color Mask
c_mask = cv2.cvtColor(res['color_mask'], cv2.COLOR_GRAY2BGR)
frames.append(create_card(c_mask, "Chromatic Color Mask", "HSV filtering for saturated colors (S >= 40, V >= 80)", 2))
durations.append(1300)

# Step 3: White Mask
w_mask = cv2.cvtColor(res['white_mask'], cv2.COLOR_GRAY2BGR)
frames.append(create_card(w_mask, "Achromatic White Mask", "HSV filtering for white stickers (S < 40, V >= 200)", 3))
durations.append(1300)

# Step 4: Cleaned Binary Mask
cl_mask = cv2.cvtColor(res['cleaned_mask'], cv2.COLOR_GRAY2BGR)
frames.append(create_card(cl_mask, "Morphological Cleaning", "Combined mask with Open (denoise) and Close (fill holes)", 4))
durations.append(1300)

# Step 5: Raw Contours Detected
raw_img = draw_contours(orig, res['raw_contours'], color=(180, 180, 180), thickness=2)
frames.append(create_card(raw_img, "Raw Contours Detected", f"Found {len(res['raw_contours'])} potential edge boundaries in cleaned mask", 5))
durations.append(1400)

# Step 6: Filtered Candidates
filt_img = draw_contours(orig, res['filtered_candidates'], color=(0, 255, 255), thickness=3)
frames.append(create_card(filt_img, "Geometric Filtering", f"Kept {len(res['filtered_candidates'])} stickers passing Area (>100px) & Rectangularity (>0.6)", 6))
durations.append(1400)

# Step 7: Final 9 Selected Stickers
sel_img = draw_contours(orig, res['selection_info']['final_group'], color=(0, 255, 0), thickness=4, label=True)
score = res['selection_info']['final_score']
frames.append(create_card(sel_img, "Optimal 9 Stickers Selected", f"Compactness score: {score:.3f} >= 0.70 threshold (Fast-Path accepted)", 7))
durations.append(1600)

# Step 8: Perspective Normalization
warp_img = res['warped']
warp_large = cv2.resize(warp_img, (480, 480), interpolation=cv2.INTER_NEAREST)
frames.append(create_card(warp_large, "Perspective Normalization", "Convex hull homography warped to 300x300 canonical square face", 8))
durations.append(1600)

# Step 9: Final Numbered Stickers & Color Swatches
final_warp = warp_img.copy()
if res['ordered_contours']:
    final_warp = draw_contours(final_warp, res['ordered_contours'], color=(0, 255, 0), thickness=3, label=True)

canvas_9 = np.zeros((460, 680, 3), dtype=np.uint8)
canvas_9[:] = (26, 32, 44)

disp_w = cv2.resize(final_warp, (380, 380), interpolation=cv2.INTER_NEAREST)
canvas_9[40:420, 20:400] = disp_w
cv2.rectangle(canvas_9, (20, 40), (400, 420), (56, 189, 248), 2)

colors = res['colors']
swatch_size = 70
gap = 8
start_x, start_y = 430, 80

for idx, bgr in enumerate(colors):
    r, c = divmod(idx, 3)
    sx = start_x + c * (swatch_size + gap)
    sy = start_y + r * (swatch_size + gap)
    b, g, r_val = [int(v) for v in bgr]
    cv2.rectangle(canvas_9, (sx, sy), (sx + swatch_size, sy + swatch_size), (b, g, r_val), -1)
    cv2.rectangle(canvas_9, (sx, sy), (sx + swatch_size, sy + swatch_size), (255, 255, 255), 1)
    tc = (0, 0, 0) if (r_val * 0.299 + g * 0.587 + b * 0.114) > 130 else (255, 255, 255)
    cv2.putText(canvas_9, f"#{idx+1}", (sx + 18, sy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tc, 2, cv2.LINE_AA)

cv2.putText(canvas_9, "Extracted 3x3 Colors", (430, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

frames.append(create_card(canvas_9, "Final Extraction & Swatches", "Ordered #1-9 stickers with sampled CIE Lab/BGR center values", 9))
durations.append(3000)

output_path = 'assets/pipeline_steps.gif'
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=False
)

print(f"Successfully generated animated GIF: {output_path} ({os.path.getsize(output_path)} bytes)")
