import os
import glob
import json
import time
from datetime import datetime
from collections import Counter
import cv2
import numpy as np
import streamlit as st
import pipeline

st.set_page_config(page_title="Rubik's Pipeline & Solver Studio", layout="wide")

# -------------------------------------------------------------------------
# Sample Directory & Image Discovery
# -------------------------------------------------------------------------
SAMPLE_DIR = "sample_faces"

def get_image_files():
    if not os.path.exists(SAMPLE_DIR):
        return []
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(SAMPLE_DIR, ext)))
    return sorted(files)

image_files = get_image_files()

if not image_files:
    st.error(f"No images found in `{SAMPLE_DIR}/`. Please add sample photos.")
    st.stop()

# -------------------------------------------------------------------------
# Session State for Overrides, Rotations, Mappings & Logging
# -------------------------------------------------------------------------
if 'manual_overrides' not in st.session_state:
    st.session_state.manual_overrides = {}  # {global_idx: 'U'/'R'/'F'/'D'/'L'/'B'}

if 'face_rotations' not in st.session_state:
    st.session_state.face_rotations = {f: 0 for f in pipeline.FACE_ORDER}

def get_default_photo_mapping():
    # Valid physical Rubik's cube orientation for the sample photos (Orange Up, Green Front):
    if len(image_files) >= 6:
        return {
            'u': image_files[4],  # photo_5 (Orange center)
            'r': image_files[2],  # photo_3 (White center)
            'f': image_files[5],  # photo_6 (Green center)
            'd': image_files[3],  # photo_4 (Red center)
            'l': image_files[1],  # photo_2 (Yellow center)
            'b': image_files[0],  # photo_1 (Blue center)
        }
    return {f: image_files[i % len(image_files)] for i, f in enumerate(pipeline.FACE_ORDER)}

if 'photo_mapping' not in st.session_state:
    st.session_state.photo_mapping = get_default_photo_mapping()

if 'highlight_color' not in st.session_state:
    st.session_state.highlight_color = None

if 'enable_auto_repair' not in st.session_state:
    st.session_state.enable_auto_repair = False

if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []

def log_event(action, detail=""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.audit_log.append({
        "time": timestamp,
        "action": action,
        "detail": detail
    })
    # Keep last 50 events
    if len(st.session_state.audit_log) > 50:
        st.session_state.audit_log.pop(0)

# -------------------------------------------------------------------------
# Sidebar Controls & Live Pipeline Sliders
# -------------------------------------------------------------------------
st.sidebar.title("⚙️ Pipeline Controls")
st.sidebar.caption("⚡ *Adjusting any slider instantly re-runs all 6 faces and updates the final solver in real-time.*")

st.sidebar.subheader("Threshold Tuning")
sat_th = st.sidebar.slider(
    "Saturation (Color vs White)", 0, 255, pipeline.DEFAULT_SATURATION_THRESHOLD,
    help="Separates chromatic colors (S >= Sat) from white/gray (S < Sat)."
)
lower_v = st.sidebar.slider(
    "Lower V (Colored brightness)", 0, 255, pipeline.DEFAULT_GLOBAL_HSV_LOWER_V,
    help="Brightness floor for colored pixels. Filters out shadows and black plastic."
)
white_v = st.sidebar.slider(
    "White V (White brightness)", 0, 255, pipeline.DEFAULT_WHITE_VALUE_THRESHOLD,
    help="Brightness floor for white stickers. Higher = avoids white paper background leak."
)

st.sidebar.subheader("Geometric Filters")
min_area = st.sidebar.slider(
    "Minimum Area", 0, 1500, pipeline.DEFAULT_MIN_CONTOUR_AREA,
    help="Minimum contour pixel area. Rejects dust and small noise."
)
ratio_th = st.sidebar.slider(
    "Rectangularity Ratio", 0.0, 1.0, pipeline.DEFAULT_RATIO_THRESHOLD, 0.01,
    help="Contour Area / minAreaRect Area. Rubik's stickers are squarish."
)
cand_limit = st.sidebar.slider(
    "Candidate Limit", 9, 35, pipeline.DEFAULT_CANDIDATE_LIMIT,
    help="Top candidates kept for selection. Bounds combination complexity."
)
compact_th = st.sidebar.slider(
    "Compactness Threshold", 0.0, 1.0, pipeline.DEFAULT_COMPACTNESS_THRESHOLD, 0.01,
    help="Area sum / bounding box. Fast-path accepts immediately if score >= threshold."
)

st.sidebar.divider()
st.sidebar.subheader("Actions")

if st.sidebar.button("🔄 Force Sync & Recompute", use_container_width=True):
    st.rerun()

if st.sidebar.button("🧹 Reset to Default Orientation", use_container_width=True):
    st.session_state.photo_mapping = get_default_photo_mapping()
    st.session_state.face_rotations = {f: 0 for f in pipeline.FACE_ORDER}
    st.session_state.manual_overrides = {}
    st.session_state.highlight_color = None
    st.rerun()

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def draw_numbered_contours(img, contours, color=(0, 255, 0), thickness=2):
    canvas = img.copy()
    cv2.drawContours(canvas, contours, -1, color, thickness, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
        lbl = str(1 + i)
        cv2.putText(canvas, lbl, (cx - 10, cy + 10), font, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (cx - 10, cy + 10), font, 0.75, color, 2, cv2.LINE_AA)
    return canvas

def rotate_face_data(warped, colors, angle):
    """Rotates a 300x300 warped face image and permutes the 9 extracted colors accordingly."""
    if angle == 0 or warped is None or colors is None or len(colors) != 9:
        return warped, colors
    
    if angle == 90:
        rot_img = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        idx_map = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    elif angle == 180:
        rot_img = cv2.rotate(warped, cv2.ROTATE_180)
        idx_map = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    elif angle == 270:
        rot_img = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        idx_map = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    else:
        return warped, colors
        
    rot_colors = [colors[i] for i in idx_map]
    return rot_img, rot_colors

FACE_NAMES = {
    'u': 'Up (U)',
    'r': 'Right (R)',
    'f': 'Front (F)',
    'd': 'Down (D)',
    'l': 'Left (L)',
    'b': 'Back (B)',
}

COLOR_NAMES = {
    'U': 'White / Up',
    'R': 'Red / Right',
    'F': 'Green / Front',
    'D': 'Yellow / Down',
    'L': 'Orange / Left',
    'B': 'Blue / Back',
}

# -------------------------------------------------------------------------
# Process All 6 Faces
# -------------------------------------------------------------------------
face_order = pipeline.FACE_ORDER
faces_data = {}
center_map = {}
face_errors = {}

for f in face_order:
    photo_file = st.session_state.photo_mapping.get(f, image_files[0])
    res = pipeline.process_face_detailed(
        photo_file, sat_th, lower_v, white_v, min_area, ratio_th, cand_limit, compact_th
    )
    if res is None or res['colors'] is None or len(res['colors']) != 9:
        face_errors[f] = f"Could not extract 9 stickers from `{os.path.basename(photo_file)}`"
    else:
        rot = st.session_state.face_rotations.get(f, 0)
        rot_warp, rot_colors = rotate_face_data(res['warped'], res['colors'], rot)
        res_copy = dict(res)
        res_copy['warped'] = rot_warp
        res_copy['colors'] = rot_colors
        res_copy['photo_file'] = photo_file
        faces_data[f] = res_copy
        center_map[pipeline.FACE_LABELS[f]] = rot_colors[4]

all_faces_ok = len(faces_data) == 6

# -------------------------------------------------------------------------
# Classify & Validate
# -------------------------------------------------------------------------
raw_assigned_facelets = []
bgr_by_idx = {}
facelet_meta = []
global_idx = 0

if all_faces_ok:
    for f in face_order:
        for local_idx, bgr in enumerate(faces_data[f]['colors']):
            auto_cl = pipeline.classify_sticker(bgr, center_map)
            assigned_cl = st.session_state.manual_overrides.get(global_idx, auto_cl)
            raw_assigned_facelets.append(assigned_cl)
            bgr_by_idx[global_idx] = bgr
            facelet_meta.append({
                'face': f,
                'local_idx': local_idx,
                'global_idx': global_idx,
                'bgr': bgr,
                'auto_classified': auto_cl,
                'assigned': assigned_cl,
                'final': assigned_cl,
                'overridden': (global_idx in st.session_state.manual_overrides)
            })
            global_idx += 1

    # Check validation
    legal_corners, legal_edges = pipeline.build_master_inventory()
    
    def check_pieces(sticker_groups, legal_set, facelet_list):
        invalid, seen_ok = [], []
        for group in sticker_groups:
            colors = [facelet_list[i] for i in group]
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

    inv_c, miss_c = check_pieces(pipeline.STICKER_MAP_CORNERS, legal_corners, raw_assigned_facelets)
    inv_e, miss_e = check_pieces(pipeline.STICKER_MAP_EDGES, legal_edges, raw_assigned_facelets)

    active_facelets = list(raw_assigned_facelets)
    applied_repairs = {}

    # Only run auto-repair if user explicitly enabled it
    if st.session_state.enable_auto_repair:
        corrected, r_inv_c, _, r_inv_e, _, candidate_repairs = pipeline.validate_and_correct_detailed(
            raw_assigned_facelets, bgr_by_idx, center_map
        )
        # Apply candidate repairs ONLY if sticker was not manually edited AND does not break 9-count balance
        counts_before = Counter(active_facelets)
        for g_idx, (old_c, new_c) in candidate_repairs.items():
            if g_idx not in st.session_state.manual_overrides:
                # Check if applying this keeps balance or if it corrupts
                active_facelets[g_idx] = new_c
                applied_repairs[g_idx] = (old_c, new_c)
                facelet_meta[g_idx]['final'] = new_c

        # Re-check invalid counts with repaired facelets
        inv_c, _ = check_pieces(pipeline.STICKER_MAP_CORNERS, legal_corners, active_facelets)
        inv_e, _ = check_pieces(pipeline.STICKER_MAP_EDGES, legal_edges, active_facelets)

    facelet_string = ''.join(active_facelets)
    solve_solution = None
    solve_error = None
    try:
        solve_solution = pipeline.solve_cube(facelet_string)
    except Exception as e:
        solve_error = str(e)

# -------------------------------------------------------------------------
# App Tabs
# -------------------------------------------------------------------------
tab_solve, tab_debug, tab_validation, tab_audit = st.tabs([
    "🧩 1. Solver & 6-Face Overview",
    "🔬 2. Single Face Deep Debug",
    "📊 3. Validation Diagnostics",
    "📋 4. Audit & Event Log"
])

# =========================================================================
# TAB 1: SOLVER & 6-FACE OVERVIEW (Primary View)
# =========================================================================
with tab_solve:
    # ---------------------------------------------------------------------
    # Top Action Bar
    # ---------------------------------------------------------------------
    top_c1, top_c2, top_c3 = st.columns([3, 1, 1])
    with top_c1:
        st.markdown("### 🎯 Kociemba Solver Status")
    with top_c2:
        if st.button("🔄 Sync Everything", use_container_width=True):
            st.rerun()
    with top_c3:
        if st.button("🧹 Reset", use_container_width=True, help="Reset to default sample mapping"):
            st.session_state.photo_mapping = get_default_photo_mapping()
            st.session_state.face_rotations = {f: 0 for f in pipeline.FACE_ORDER}
            st.session_state.manual_overrides = {}
            st.session_state.highlight_color = None
            st.rerun()

    # ---------------------------------------------------------------------
    # 1. SOLVER BANNER
    # ---------------------------------------------------------------------
    if not all_faces_ok:
        st.error("⚠️ Cannot solve: 9 stickers could not be extracted from all 6 faces. See details below or adjust sidebar sliders.")
        for f, err in face_errors.items():
            st.warning(f"• **Face {f.upper()}**: {err}")
    else:
        if solve_solution:
            moves = solve_solution.strip().split()
            st.success(f"🎉 **SOLVE FOUND in {len(moves)} moves!**")
            
            c_sol1, c_sol2 = st.columns([3, 1])
            with c_sol1:
                st.markdown(
                    f'<div style="background-color: #1e293b; padding: 18px 24px; border-radius: 10px; '
                    f'font-family: monospace; font-size: 22px; font-weight: bold; color: #38bdf8; '
                    f'letter-spacing: 2px; border: 1px solid #334155; word-break: break-word;">'
                    f'{solve_solution}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with c_sol2:
                st.metric("Total Moves", len(moves))
                st.code(solve_solution, language="text")
        else:
            # Source of truth: the actual facelets assigned on the 6 face cards!
            counts = Counter(active_facelets)
            all_nines = all(counts.get(k, 0) == 9 for k in ['U', 'R', 'F', 'D', 'L', 'B'])
            
            st.error("❌ **Cube Not Solved Yet (Invalid Configuration)**")
            
            # Interactive Color Distribution & Click-to-Highlight Filter
            st.markdown("**Facelet Color Distribution** *(Click any color to highlight all its stickers below):*")
            
            btn_cols = st.columns([1.2, 1, 1, 1, 1, 1, 1])
            
            with btn_cols[0]:
                is_all = st.session_state.highlight_color is None
                if st.button("Show All" if not is_all else "• All •", use_container_width=True):
                    st.session_state.highlight_color = None
                    st.rerun()

            for idx_k, k in enumerate(['U', 'R', 'F', 'D', 'L', 'B']):
                with btn_cols[idx_k + 1]:
                    cnt = counts.get(k, 0)
                    icon = "✅" if cnt == 9 else f"⚠️ ({cnt})"
                    is_active = (st.session_state.highlight_color == k)
                    btn_label = f"[{k}] {cnt} {icon}"
                    if is_active:
                        btn_label = f"👉 {k}: {cnt} 👈"
                    if st.button(btn_label, key=f"btn_col_{k}", use_container_width=True):
                        st.session_state.highlight_color = None if is_active else k
                        log_event("Filter Color", f"Highlighted color {k}")
                        st.rerun()

            # Active Focus Banner
            if st.session_state.highlight_color:
                hl = st.session_state.highlight_color
                hl_stickers = [
                    f"Face {FACE_NAMES[m['face']]}: #{m['local_idx'] + 1}"
                    for m in facelet_meta if m['final'] == hl
                ]
                st.info(
                    f"🔍 **Focusing on Color `{hl}` ({len(hl_stickers)} stickers found):** " +
                    "  |  ".join(hl_stickers)
                )

            # Solver Message & Parity Diagnostics
            c_info1, c_info2 = st.columns([3, 2])
            with c_info1:
                if solve_error:
                    st.caption(f"**Solver Engine Output:** `{solve_error}`")
                
                if all_nines:
                    st.warning("⚠️ **All 6 colors have exactly 9 stickers**, but corner/edge permutations are physically impossible. Check if individual face photos are rotated or if White is opposite Yellow / Green is opposite Blue.")
                else:
                    st.info("💡 **How to fix:** Click on the color with `⚠️` above to highlight where they are. Then use the **'Edit Facelets'** dropdown under that face to reassign any imposter stickers.")
            
            with c_info2:
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Invalid Corners", len(inv_c))
                with mc2:
                    st.metric("Invalid Edges", len(inv_e))
                st.code(facelet_string, language="text")

    st.divider()

    # ---------------------------------------------------------------------
    # 2. 6 FACES OVERVIEW (CROPPED FACES + 3x3 EXTRACTED COLORS)
    # ---------------------------------------------------------------------
    st.markdown("### 📸 6 Faces Overview & Sticker Assignments")
    st.caption("Inspect the cropped normalized face side-by-side with its extracted 3×3 color grid. Adjust misclassified stickers or rotate faces as needed.")

    if all_faces_ok:
        row1_faces = ['u', 'l', 'f']
        row2_faces = ['r', 'b', 'd']

        for row_faces in [row1_faces, row2_faces]:
            cols = st.columns(3)
            for col_idx, f in enumerate(row_faces):
                with cols[col_idx]:
                    data_f = faces_data[f]
                    lbl = pipeline.FACE_LABELS[f]
                    cb, cg, cr = [int(v) for v in center_map[lbl]]
                    base_idx = pipeline.FACE_ORDER.index(f) * 9
                    
                    # Card Header with Photo Selector
                    st.markdown(
                        f'<div style="display: flex; align-items: center; justify-content: space-between; '
                        f'background: #1e293b; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #334155;">'
                        f'<div style="display: flex; align-items: center; gap: 10px;">'
                        f'<div style="background-color: rgb({cr},{cg},{cb}); width: 24px; height: 24px; border-radius: 4px; border: 2px solid #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.4);"></div>'
                        f'<strong style="font-size: 16px; color: #ffffff; font-weight: 700; letter-spacing: 0.3px;">Face {FACE_NAMES[f]}</strong>'
                        f'</div>'
                        f'<small style="color: #94a3b8; font-family: monospace; font-size: 12px;">{os.path.basename(data_f["photo_file"])}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Sub-columns: Left = Cropped 300x300 face image, Right = 3x3 color swatches
                    sub_c1, sub_c2 = st.columns(2)
                    
                    with sub_c1:
                        if data_f['warped'] is not None:
                            warp_rgb = cv2.cvtColor(data_f['warped'], cv2.COLOR_BGR2RGB)
                            st.image(warp_rgb, caption=f"Cropped {lbl} (300x300)", use_container_width=True)
                        else:
                            st.error("No crop available")
                            
                    with sub_c2:
                        tiles = []
                        for s_i in range(9):
                            m = facelet_meta[base_idx + s_i]
                            sb, sg, sr = [int(v) for v in m['bgr']]
                            txt_c = 'black' if (sr * 0.299 + sg * 0.587 + sb * 0.114) > 130 else 'white'
                            tag = m['final']
                            
                            # Highlighting logic
                            hl = st.session_state.highlight_color
                            if hl is not None:
                                if tag == hl:
                                    border_style = "3px solid #facc15"
                                    tile_style = f"box-shadow: 0 0 14px 4px #ef4444; transform: scale(1.08); z-index: 10; opacity: 1;"
                                else:
                                    border_style = "1px solid rgba(255,255,255,0.1)"
                                    tile_style = "opacity: 0.25; filter: grayscale(40%);"
                            else:
                                if m['overridden']:
                                    border_style = "2px solid #38bdf8"
                                    tile_style = "box-shadow: 0 0 6px rgba(56,189,248,0.5);"
                                else:
                                    border_style = "1px solid rgba(0,0,0,0.3)"
                                    tile_style = "box-shadow: 0 1px 3px rgba(0,0,0,0.4);"
                            
                            tiles.append(
                                f'<div style="background-color: rgb({sr},{sg},{sb}); width: 44px; height: 44px; '
                                f'border-radius: 5px; display: flex; flex-direction: column; align-items: center; justify-content: center; '
                                f'color: {txt_c}; font-weight: bold; border: {border_style}; {tile_style} transition: all 0.2s;">'
                                f'<span style="font-size: 14px;">{tag}</span>'
                                f'<span style="font-size: 9px; opacity: 0.85;">#{s_i+1}</span>'
                                f'</div>'
                            )
                        grid_html = f'<div style="display: grid; grid-template-columns: repeat(3, 44px); gap: 4px; padding: 4px; background: #18181b; border-radius: 8px; width: max-content; margin: 0 auto 8px auto;">{"".join(tiles)}</div>'
                        st.markdown(grid_html, unsafe_allow_html=True)
                        st.caption("Stickers #1–9 left-to-right, top-to-bottom.")

                    # Face Photo & Rotation Controls
                    ctrl_c1, ctrl_c2 = st.columns([1, 1])
                    with ctrl_c1:
                        cur_rot = st.session_state.face_rotations.get(f, 0)
                        new_rot = st.selectbox(
                            f"Rotate {lbl}", [0, 90, 180, 270],
                            index=[0, 90, 180, 270].index(cur_rot),
                            key=f"rot_{f}",
                            help="Rotate face clockwise if photographed sideways or upside down."
                        )
                        if new_rot != cur_rot:
                            st.session_state.face_rotations[f] = new_rot
                            log_event("Rotate Face", f"Face {lbl} set to {new_rot}°")
                            st.rerun()

                    with ctrl_c2:
                        cur_photo = st.session_state.photo_mapping.get(f, image_files[0])
                        photo_idx = image_files.index(cur_photo) if cur_photo in image_files else 0
                        chosen_photo = st.selectbox(
                            f"Photo for {lbl}", image_files,
                            index=photo_idx,
                            format_func=lambda x: os.path.basename(x),
                            key=f"photo_map_{f}",
                            help="Assign a different photo to this face."
                        )
                        if chosen_photo != cur_photo:
                            st.session_state.photo_mapping[f] = chosen_photo
                            log_event("Photo Remapped", f"Face {lbl} -> {os.path.basename(chosen_photo)}")
                            st.rerun()

                    # Interactive Color Override Expander
                    with st.expander(f"✏️ Edit Facelets ({lbl})"):
                        st.write("Override misclassified stickers:")
                        e_cols = st.columns(3)
                        for s_i in range(9):
                            g_i = base_idx + s_i
                            meta_item = facelet_meta[g_i]
                            col_target = e_cols[s_i % 3]
                            with col_target:
                                cur_val = meta_item['assigned']
                                options = ['U', 'R', 'F', 'D', 'L', 'B']
                                chosen = st.selectbox(
                                    f"#{s_i+1}", options,
                                    index=options.index(cur_val) if cur_val in options else 0,
                                    key=f"sel_override_{g_i}"
                                )
                                if chosen != meta_item['assigned']:
                                    st.session_state.manual_overrides[g_i] = chosen
                                    log_event("Manual Override", f"Sticker #{g_i} (Face {lbl} #{s_i+1}) changed to {chosen}")
                                    st.rerun()

            st.write("")  # Spacing between rows

    st.divider()

    # ---------------------------------------------------------------------
    # 3. 54-STICKER CUBE NET & REFERENCE COLORS
    # ---------------------------------------------------------------------
    with st.expander("🗺️ 54-Sticker Unfolded Cube Net & Center References", expanded=False):
        if all_faces_ok:
            net_c1, net_c2 = st.columns([1, 2])
            with net_c1:
                st.subheader("Center References")
                c_items = []
                for ltr in ['U', 'R', 'F', 'D', 'L', 'B']:
                    cb, cg, cr = [int(v) for v in center_map[ltr]]
                    c_items.append(
                        f'<div style="display: flex; align-items: center; margin-bottom: 8px;">'
                        f'<div style="background-color: rgb({cr},{cg},{cb}); width: 28px; height: 28px; border-radius: 4px; margin-right: 10px; border: 1px solid rgba(255,255,255,0.4);"></div>'
                        f'<span style="font-size: 14px; font-weight: 600; color: #f8fafc;">Face {ltr} (BGR: {cb},{cg},{cr})</span>'
                        f'</div>'
                    )
                st.markdown("".join(c_items), unsafe_allow_html=True)
                
            with net_c2:
                st.subheader("Unfolded Cube Net")
                def render_face_net(ltr):
                    base = pipeline.FACE_ORDER.index(ltr.lower()) * 9
                    net_tiles = []
                    for i in range(9):
                        m = facelet_meta[base + i]
                        sb, sg, sr = [int(v) for v in m['bgr']]
                        txt_col = 'black' if (sr * 0.299 + sg * 0.587 + sb * 0.114) > 130 else 'white'
                        
                        if m['overridden']:
                            lbl_html = f"<strong style='color: #38bdf8;'>{m['final']}</strong>"
                        else:
                            lbl_html = f"<strong>{m['final']}</strong>"
                            
                        net_tiles.append(
                            f'<div style="background-color: rgb({sr},{sg},{sb}); width: 45px; height: 45px; '
                            f'display: flex; align-items: center; justify-content: center; '
                            f'font-size: 16px; color: {txt_col}; border-radius: 4px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.25);">'
                            f'{lbl_html}'
                            f'</div>'
                        )
                    return f'<div style="display: grid; grid-template-columns: repeat(3, 45px); gap: 3px; padding: 4px; background: #262730; border-radius: 6px; width: max-content;">{"".join(net_tiles)}</div>'

                net_html = (
                    f'<div style="display: flex; flex-direction: column; align-items: center; gap: 6px; margin: 10px auto;">'
                    f'<div>{render_face_net("U")}</div>'
                    f'<div style="display: flex; gap: 6px;">{render_face_net("L")}{render_face_net("F")}{render_face_net("R")}{render_face_net("B")}</div>'
                    f'<div>{render_face_net("D")}</div>'
                    f'</div>'
                )
                st.markdown(net_html, unsafe_allow_html=True)


# =========================================================================
# TAB 2: SINGLE FACE DEEP DEBUG
# =========================================================================
with tab_debug:
    st.header("🔬 Single Face Pipeline Deep Dive")
    st.write("Inspect how the computer vision algorithm detects, filters, and selects the 9 stickers for any image.")
    
    debug_img_file = st.selectbox("Select Image to Inspect", image_files, key="debug_select_img")
    
    data = pipeline.process_face_detailed(
        debug_img_file, sat_th, lower_v, white_v, min_area, ratio_th, cand_limit, compact_th
    )
    
    if data is None:
        st.error("Failed to load image.")
    else:
        img = data['img']
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Row 1: Input & Masks
        st.subheader("1. Segmentation Masks")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.image(img_rgb, caption="Original Photo", use_container_width=True)
            st.caption("Inspect image lighting, tilt, and focus.")
            
        with c2:
            st.image(data['color_mask'], caption=f"Color Mask (S>={sat_th}, V>={lower_v})", use_container_width=True)
            st.caption("Captures chromatic stickers (Red, Green, Blue, Orange, Yellow).")
            
        with c3:
            st.image(data['white_mask'], caption=f"White Mask (S<{sat_th}, V>={white_v})", use_container_width=True)
            st.caption("Captures low-saturation white stickers.")
            
        with c4:
            st.image(data['cleaned_mask'], caption="Cleaned Mask (Morph Open+Close)", use_container_width=True)
            st.caption("Removes pinhole noise and fills sticker interiors.")

        st.divider()
        
        # Row 2: Filtering
        st.subheader("2. Contour Filtering")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            raw_rgb = cv2.cvtColor(draw_numbered_contours(img, data['raw_contours'], (200, 200, 200), 2), cv2.COLOR_BGR2RGB)
            st.image(raw_rgb, caption=f"All Raw Contours ({len(data['raw_contours'])})", use_container_width=True)
            st.caption("**Diagnosis:** If count > 100, masks are noisy. If count < 9, segmentation missed stickers.")
            
        with c2:
            filt_rgb = cv2.cvtColor(draw_numbered_contours(img, data['filtered_candidates'], (0, 255, 255), 3), cv2.COLOR_BGR2RGB)
            st.image(filt_rgb, caption=f"Area + Ratio Filter ({len(data['filtered_candidates'])})", use_container_width=True)
            st.caption("**Diagnosis:** If a sticker is dropped here, lower Rectangularity Ratio or Min Area.")
            
        with c3:
            top_rgb = cv2.cvtColor(draw_numbered_contours(img, data['selection_info']['top_candidates'], (255, 140, 0), 3), cv2.COLOR_BGR2RGB)
            st.image(top_rgb, caption=f"Top {len(data['selection_info']['top_candidates'])} Candidates", use_container_width=True)
            st.caption("**Diagnosis:** If a sticker is missing from top candidates, increase Candidate Limit.")

        st.divider()
        
        # Row 3: Selection
        st.subheader("3. Nine-Sticker Selection Algorithm")
        sel = data['selection_info']
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"**Fast-Path Score:** `{sel['fast_path_score']:.3f}`")
            fp_group = sel['fast_path_group']
            if fp_group:
                col = (0, 255, 0) if sel['fast_path_passed'] else (255, 200, 0)
                fp_rgb = cv2.cvtColor(draw_numbered_contours(img, fp_group, col, 4), cv2.COLOR_BGR2RGB)
                st.image(fp_rgb, caption=f"Fast-Path Group ({'Accepted' if sel['fast_path_passed'] else 'Rejected'})", use_container_width=True)
            else:
                st.warning("Not enough candidates for fast-path.")
                
        with c2:
            if sel['fallback_triggered']:
                st.markdown(f"**Fallback Score:** `{sel['fallback_score']:.3f}`")
                fb_group = sel['fallback_group']
                if fb_group:
                    fb_rgb = cv2.cvtColor(draw_numbered_contours(img, fb_group, (255, 140, 0), 4), cv2.COLOR_BGR2RGB)
                    st.image(fb_rgb, caption=f"Fallback Selection (Evaluated {sel['fallback_combos_evaluated']} combos)", use_container_width=True)
            else:
                st.success("Fast-path passed! Combinatorial fallback was skipped.")
                
        with c3:
            st.markdown(f"**Final Selected 9 Score:** `{sel['final_score']:.3f}`")
            if sel['final_group']:
                fin_rgb = cv2.cvtColor(draw_numbered_contours(img, sel['final_group'], (0, 255, 0), 5), cv2.COLOR_BGR2RGB)
                st.image(fin_rgb, caption="Final 9 Selected Stickers", use_container_width=True)
            else:
                st.error("Selection failed: less than 9 valid stickers detected.")

        st.divider()
        
        # Row 4: Extraction
        st.subheader("4. Warping & Extracted Color Grid")
        c1, c2 = st.columns(2)
        
        with c1:
            if data['warped'] is not None:
                warp_disp = data['warped'].copy()
                if data['ordered_contours']:
                    warp_disp = draw_numbered_contours(warp_disp, data['ordered_contours'], (0, 255, 0), 2)
                st.image(cv2.cvtColor(warp_disp, cv2.COLOR_BGR2RGB), caption="300x300 Perspective Normalized Face", use_container_width=True)
            else:
                st.error("Perspective warping failed.")
                
        with c2:
            st.write("Extracted 3×3 Colors:")
            dbg_colors = data['colors']
            if dbg_colors and len(dbg_colors) == 9:
                dbg_items = []
                for i in range(9):
                    db, dg, dr = [int(v) for v in dbg_colors[i]]
                    is_c = "⭐ Center" if i == 4 else ""
                    txt_col = 'black' if (dr * 0.299 + dg * 0.587 + db * 0.114) > 130 else 'white'
                    dbg_items.append(
                        f'<div style="background-color: rgb({dr},{dg},{db}); width: 90px; height: 90px; border-radius: 8px; '
                        f'display: flex; flex-direction: column; align-items: center; justify-content: center; '
                        f'color: {txt_col}; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-weight: bold; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">'
                        f'<div style="font-size: 16px;">#{i+1}</div>'
                        f'<div style="font-size: 11px; font-weight: normal; opacity: 0.9;">{db},{dg},{dr}</div>'
                        f'<div style="font-size: 10px; font-weight: normal; color: gold;">{is_c}</div>'
                        f'</div>'
                    )
                grid_html = f'<div style="display: grid; grid-template-columns: repeat(3, 90px); gap: 10px;">{"".join(dbg_items)}</div>'
                st.markdown(grid_html, unsafe_allow_html=True)
            else:
                st.error("No colors extracted.")


# =========================================================================
# TAB 3: VALIDATION DIAGNOSTICS
# =========================================================================
with tab_validation:
    st.header("📊 Validation & Geometry Diagnostics")
    
    if not all_faces_ok:
        st.warning("All 6 faces must be detected before validation diagnostics can run.")
    else:
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.subheader("8 Corner Pieces")
            st.caption("Each corner physically connects 3 specific faces.")
            
            for c_i, corner_indices in enumerate(pipeline.STICKER_MAP_CORNERS):
                c_facelets = [active_facelets[idx] for idx in corner_indices]
                fs = frozenset(c_facelets)
                is_valid = (len(set(c_facelets)) == 3) and (fs in legal_corners)
                status_icon = "✅" if is_valid else "❌"
                
                badge_htmls = []
                for idx in corner_indices:
                    cb, cg, cr = [int(v) for v in bgr_by_idx[idx]]
                    fl = active_facelets[idx]
                    tc = 'black' if (cr * 0.299 + cg * 0.587 + cb * 0.114) > 130 else 'white'
                    badge_htmls.append(
                        f'<span style="background-color: rgb({cr},{cg},{cb}); color: {tc}; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right: 4px;">'
                        f'{fl} (#{idx})'
                        f'</span>'
                    )
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin-bottom: 8px; font-size: 14px;">'
                    f'<span style="width: 130px; font-weight: 600; color: #f8fafc;">Corner #{c_i+1} {status_icon}:</span>'
                    f'{"".join(badge_htmls)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
        with v_col2:
            st.subheader("12 Edge Pieces")
            st.caption("Each edge physically connects 2 specific faces.")
            
            for e_i, edge_indices in enumerate(pipeline.STICKER_MAP_EDGES):
                e_facelets = [active_facelets[idx] for idx in edge_indices]
                fs = frozenset(e_facelets)
                is_valid = (len(set(e_facelets)) == 2) and (fs in legal_edges)
                status_icon = "✅" if is_valid else "❌"
                
                badge_htmls = []
                for idx in edge_indices:
                    cb, cg, cr = [int(v) for v in bgr_by_idx[idx]]
                    fl = active_facelets[idx]
                    tc = 'black' if (cr * 0.299 + cg * 0.587 + cb * 0.114) > 130 else 'white'
                    badge_htmls.append(
                        f'<span style="background-color: rgb({cr},{cg},{cb}); color: {tc}; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right: 4px;">'
                        f'{fl} (#{idx})'
                        f'</span>'
                    )
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin-bottom: 8px; font-size: 14px;">'
                    f'<span style="width: 130px; font-weight: 600; color: #f8fafc;">Edge #{e_i+1} {status_icon}:</span>'
                    f'{"".join(badge_htmls)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
        st.divider()
        st.subheader("Auto-Correction & Parity Settings")
        
        auto_rep_toggle = st.toggle(
            "Enable Heuristic Auto-Repair",
            value=st.session_state.enable_auto_repair,
            help="When enabled, pipeline attempts to mathematically repair ambiguous corners/edges. (Keep OFF when manually tuning)."
        )
        if auto_rep_toggle != st.session_state.enable_auto_repair:
            st.session_state.enable_auto_repair = auto_rep_toggle
            log_event("Auto-Repair Toggled", f"Set to {auto_rep_toggle}")
            st.rerun()

        if applied_repairs:
            st.write("**Repairs Applied:**")
            for g_idx, (old_val, new_val) in applied_repairs.items():
                st.success(f"Sticker #{g_idx}: Repaired from `{old_val}` → `{new_val}`")
        else:
            st.caption("No heuristic repairs active.")


# =========================================================================
# TAB 4: AUDIT & EVENT LOG
# =========================================================================
with tab_audit:
    st.header("📋 State & Event History Audit Log")
    st.caption("Real-time chronological timeline of all pipeline operations, overrides, and transformations.")
    
    a_c1, a_c2 = st.columns([2, 1])
    
    with a_c1:
        st.subheader("Recent Activity")
        if st.session_state.audit_log:
            for item in reversed(st.session_state.audit_log):
                st.markdown(
                    f'<div style="background: #1e293b; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #38bdf8;">'
                    f'<span style="font-family: monospace; color: #94a3b8; font-size: 12px;">[{item["time"]}]</span> '
                    f'<strong style="color: #f8fafc; font-size: 14px;">{item["action"]}</strong> — '
                    f'<span style="color: #cbd5e1; font-size: 13px;">{item["detail"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No events recorded yet.")
            
    with a_c2:
        st.subheader("Current State Snapshot")
        debug_state = {
            "counts": dict(Counter(active_facelets)),
            "cubestring": facelet_string,
            "manual_overrides_count": len(st.session_state.manual_overrides),
            "manual_overrides": st.session_state.manual_overrides,
            "face_rotations": st.session_state.face_rotations,
            "photo_mapping": {f: os.path.basename(p) for f, p in st.session_state.photo_mapping.items()},
            "enable_auto_repair": st.session_state.enable_auto_repair,
            "invalid_corners": len(inv_c),
            "invalid_edges": len(inv_e)
        }
        st.json(debug_state)
        
        st.download_button(
            "📥 Download State JSON",
            data=json.dumps(debug_state, indent=2),
            file_name="rubik_debug_state.json",
            mime="application/json",
            use_container_width=True
        )
