"""
D435 branch detection + STM32 curl — NO drone/flight control.
Use this to test the camera + arm working together without any Pixhawk.
"""
import cv2
import numpy as np
import os
import select
import serial
import sys
import threading
import time

HEADLESS = not os.environ.get("DISPLAY")
if HEADLESS:
    import yaml
    print("No display detected — running headless (no GUI windows)")
else:
    from param_gui import ParamGUI

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise ImportError(
        "This script requires `pyrealsense2` for the Intel RealSense D435 camera. "
        "Install dependencies with: pip install numpy opencv-python pyrealsense2 pyyaml"
    ) from exc

# -----------------------------
# Stream profiles to try
# -----------------------------
STREAM_CANDIDATES = [
    {
        "color_width": 640,
        "color_height": 480,
        "color_fps": 15,
        "depth_width": 640,
        "depth_height": 480,
        "depth_fps": 15,
    },
    {
        "color_width": 640,
        "color_height": 480,
        "color_fps": 15,
        "depth_width": 848,
        "depth_height": 480,
        "depth_fps": 15,
    },
]

# -----------------------------
# Depth tuning
# -----------------------------
MIN_DEPTH_M = 0.04
MAX_DEPTH_M = 2.00
GREEN_THRESHOLD_M = 0.20

MIN_DEPTH_MM = int(MIN_DEPTH_M * 1000)
MAX_DEPTH_MM = int(MAX_DEPTH_M * 1000)

# -----------------------------
# Main Hough-line detector region
# -----------------------------
TOP_REGION_FRACTION = 2.0 / 2.0

# -----------------------------
# RGB preprocessing
# -----------------------------
CANNY_LOW = 19
CANNY_HIGH = 69
GAUSSIAN_KERNEL = (11, 11)

CLAHE_CLIP_LIMIT = 6.2
CLAHE_TILE_GRID = (8, 8)

MORPH_CLOSE_KERNEL = (13, 5)
MORPH_OPEN_KERNEL = (5, 3)

MASK_ERODE_KERNEL = (5, 3)

# -----------------------------
# Hough-line / line-pair filters
# -----------------------------
HOUGH_THRESHOLD = 131
HOUGH_MIN_LINE_LENGTH = 142
HOUGH_MIN_LINE_LENGTH_REFERENCE_DEPTH_M = 1.0
HOUGH_LINE_CONNECT_GAP_REFERENCE_PX = 28
HOUGH_LINE_CONNECT_GAP_REFERENCE_DEPTH_M = 1.0
HOUGH_LINE_CONNECT_GAP_MIN_PX = 105
HOUGH_LINE_CONNECT_GAP_MAX_PX = 206

MAX_HORIZONTAL_DEVIATION_DEG = 30.0
MAX_PAIR_ANGLE_DIFF_DEG = 24.0
MIN_ESTIMATED_LENGTH_M = 0.127  # 5 inches
MIN_PAIR_GAP_M = 0.0254  # 1 inch
MAX_PAIR_GAP_M = 0.0889  # 3.5 inches
MIN_MEASURABLE_PAIR_GAP_PX = 4.0
MERGE_LINE_ANGLE_DIFF_DEG = 4.0
MERGE_LINE_OFFSET_PX = 8.0
MERGE_MIN_OVERLAP_RATIO = 0.60
PAIR_PAD_X = 10
PAIR_PAD_Y = 6

# -----------------------------
# Depth voting
# -----------------------------
DEPTH_BIN_MM = 20
MIN_VALID_DEPTH_RATIO = 0.05
MIN_MAJORITY_RATIO = 0.18
MAX_BACKGROUND_RATIO = 0.60
BACKGROUND_DEPTH_TOLERANCE_BINS = 2.0

# -----------------------------
# Display / debugging
# -----------------------------
DRAW_TOP_REGION_LINE = True
TOP_REGION_COLOR = (255, 255, 0)
BOX_RED = (0, 0, 255)
BOX_YELLOW = (0, 255, 255)
BOX_GREEN = (0, 255, 0)
YELLOW_HORIZONTAL_DEV_DEG = 15.0
MAIN_WINDOW_NAME = "D435 Branch Detection (no drone)"
TRACK_SMOOTHING_ALPHA = 0.35
TRACK_MAX_MISSING_FRAMES = 4
TRACK_MATCH_MAX_CENTER_PX = 120.0
TRACK_MATCH_MIN_IOU = 0.05
DEBUG_IMAGE_ALPHA = 0.35
HOUGH_LINE_LINGER_SECONDS = 0.35
HOUGH_LINE_COMPUTE_LINGER_SECONDS = 0.30


def safe_set_option(sensor_or_filter, option, value):
    try:
        sensor_or_filter.set_option(option, value)
        return True
    except Exception:
        return False


def apply_camera_runtime_params(
    depth_sensor,
    spatial_filter,
    temporal_filter,
    runtime_params,
    last_camera_settings,
):
    settings_to_apply = {
        "visual_preset": float(runtime_params["visual_preset"]),
        "emitter_enabled": float(runtime_params["emitter_enabled"]),
        "laser_power": float(runtime_params["laser_power"]),
        "auto_exposure": float(runtime_params["auto_exposure"]),
        "exposure": float(runtime_params["exposure"]),
        "gain": float(runtime_params["gain"]),
        "spatial_magnitude": float(runtime_params["spatial_magnitude"]),
        "spatial_alpha": float(runtime_params["spatial_alpha"]),
        "spatial_delta": float(runtime_params["spatial_delta"]),
        "temporal_alpha": float(runtime_params["temporal_alpha"]),
        "temporal_delta": float(runtime_params["temporal_delta"]),
    }

    option_targets = {
        "visual_preset": (depth_sensor, rs.option.visual_preset),
        "emitter_enabled": (depth_sensor, rs.option.emitter_enabled),
        "laser_power": (depth_sensor, rs.option.laser_power),
        "auto_exposure": (depth_sensor, rs.option.enable_auto_exposure),
        "exposure": (depth_sensor, rs.option.exposure),
        "gain": (depth_sensor, rs.option.gain),
        "spatial_magnitude": (spatial_filter, rs.option.filter_magnitude),
        "spatial_alpha": (spatial_filter, rs.option.filter_smooth_alpha),
        "spatial_delta": (spatial_filter, rs.option.filter_smooth_delta),
        "temporal_alpha": (temporal_filter, rs.option.filter_smooth_alpha),
        "temporal_delta": (temporal_filter, rs.option.filter_smooth_delta),
    }

    for key, value in settings_to_apply.items():
        if last_camera_settings.get(key) == value:
            continue
        target, option = option_targets[key]
        if safe_set_option(target, option, value):
            last_camera_settings[key] = value


def start_pipeline_with_fallback():
    pipeline = rs.pipeline()

    for candidate in STREAM_CANDIDATES:
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            candidate["color_width"],
            candidate["color_height"],
            rs.format.bgr8,
            candidate["color_fps"],
        )
        config.enable_stream(
            rs.stream.depth,
            candidate["depth_width"],
            candidate["depth_height"],
            rs.format.z16,
            candidate["depth_fps"],
        )

        try:
            profile = pipeline.start(config)
            print("Started stream profile:")
            print(candidate)
            return pipeline, profile, candidate
        except RuntimeError as exc:
            print(f"Stream profile failed: {candidate}")
            print(f"Reason: {exc}")

    raise RuntimeError("No supported stream combination could be started.")


def horizontal_deviation_degrees(angle):
    return min(angle, abs(180.0 - angle))


def line_angle_degrees(x1, y1, x2, y2):
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle % 180.0


def angle_difference_degrees(angle_a, angle_b):
    difference = abs(angle_a - angle_b)
    return min(difference, 180.0 - difference)


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def longest_edge_angle_deg(box_points):
    best_len = -1.0
    best_angle = 0.0

    for i in range(4):
        p1 = box_points[i]
        p2 = box_points[(i + 1) % 4]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        edge_len = np.hypot(dx, dy)
        if edge_len > best_len:
            best_len = edge_len
            best_angle = np.degrees(np.arctan2(dy, dx)) % 180.0

    return best_angle


def line_length(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def line_midpoint(line):
    x1, y1, x2, y2 = line
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def line_direction(line):
    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = np.hypot(dx, dy)
    if length <= 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([dx / length, dy / length], dtype=np.float32)


def pair_direction(line_a, line_b):
    direction = line_direction(line_a) + line_direction(line_b)
    norm = np.hypot(direction[0], direction[1])
    if norm <= 1e-6:
        return line_direction(line_a)
    return direction / norm


def line_projection_overlap_ratio(line_a, line_b):
    direction = pair_direction(line_a, line_b)
    a_points = [(line_a[0], line_a[1]), (line_a[2], line_a[3])]
    b_points = [(line_b[0], line_b[1]), (line_b[2], line_b[3])]

    a_proj = [float(np.dot(point, direction)) for point in a_points]
    b_proj = [float(np.dot(point, direction)) for point in b_points]

    a_min, a_max = min(a_proj), max(a_proj)
    b_min, b_max = min(b_proj), max(b_proj)
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    min_length = min(a_max - a_min, b_max - b_min)
    if min_length <= 0:
        return 0.0
    return overlap / min_length


def line_pair_gap_px(line_a, line_b):
    direction = pair_direction(line_a, line_b)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    return abs(float(np.dot(line_midpoint(line_b) - line_midpoint(line_a), normal)))


def merge_similar_hough_lines(lines):
    merged = []

    for line in sorted(lines, key=lambda segment: line_length(*segment), reverse=True):
        line_angle = line_angle_degrees(*line)
        is_duplicate = False

        for existing in merged:
            angle_diff = angle_difference_degrees(line_angle, existing["angle_deg"])
            if angle_diff > MERGE_LINE_ANGLE_DIFF_DEG:
                continue

            offset_px = line_pair_gap_px(line, existing["segment"])
            if offset_px > MERGE_LINE_OFFSET_PX:
                continue

            overlap_ratio = line_projection_overlap_ratio(line, existing["segment"])
            if overlap_ratio < MERGE_MIN_OVERLAP_RATIO:
                continue

            is_duplicate = True
            break

        if is_duplicate:
            continue

        merged.append(
            {
                "segment": line,
                "angle_deg": line_angle,
                "horizontal_dev_deg": horizontal_deviation_degrees(line_angle),
                "length_px": line_length(*line),
            }
        )

    return merged


def estimate_length_m(pixel_length, depth_m, focal_length_px):
    if focal_length_px <= 0:
        return None
    return float(pixel_length) * float(depth_m) / float(focal_length_px)


def estimate_pixels_for_length(length_m, depth_m, focal_length_px):
    if depth_m <= 0:
        return None
    return float(length_m) * float(focal_length_px) / float(depth_m)


def build_hough_gap_schedule(focal_length_px, runtime_params):
    reference_gap_m = estimate_length_m(
        runtime_params["gap_ref_px"],
        HOUGH_LINE_CONNECT_GAP_REFERENCE_DEPTH_M,
        focal_length_px,
    )
    if reference_gap_m is None:
        return [runtime_params["gap_ref_px"]]

    depth_samples_m = [
        runtime_params["min_depth_m"],
        runtime_params["green_threshold_m"],
        0.35, 0.50, 0.75, 1.00,
        runtime_params["max_depth_m"],
    ]

    gap_schedule = []
    for depth_m in depth_samples_m:
        depth_m = float(np.clip(depth_m, runtime_params["min_depth_m"], runtime_params["max_depth_m"]))
        gap_px = estimate_pixels_for_length(reference_gap_m, depth_m, focal_length_px)
        if gap_px is None:
            continue
        gap_px = int(round(np.clip(gap_px, runtime_params["gap_min_px"], runtime_params["gap_max_px"])))
        gap_schedule.append(gap_px)

    gap_schedule.append(runtime_params["gap_ref_px"])
    return sorted(set(gap_schedule))


def build_hough_min_length_schedule(focal_length_px, runtime_params):
    reference_length_m = estimate_length_m(
        runtime_params["hough_min_line_length"],
        HOUGH_MIN_LINE_LENGTH_REFERENCE_DEPTH_M,
        focal_length_px,
    )
    if reference_length_m is None:
        return [runtime_params["hough_min_line_length"]]

    depth_samples_m = [
        runtime_params["min_depth_m"],
        runtime_params["green_threshold_m"],
        0.35, 0.50, 0.75, 1.00,
        runtime_params["max_depth_m"],
    ]

    min_length_schedule = []
    for depth_m in depth_samples_m:
        depth_m = float(np.clip(depth_m, runtime_params["min_depth_m"], runtime_params["max_depth_m"]))
        min_length_px = estimate_pixels_for_length(reference_length_m, depth_m, focal_length_px)
        if min_length_px is None:
            continue
        min_length_px = int(round(max(1, min_length_px)))
        min_length_schedule.append(min_length_px)

    min_length_schedule.append(runtime_params["hough_min_line_length"])
    return sorted(set(min_length_schedule))


def line_has_in_range_depth(depth_image, line, runtime_params):
    x1, y1, x2, y2 = line
    image_h, image_w = depth_image.shape[:2]
    sample_count = int(np.clip(round(line_length(*line) / 3.0), 8, 160))

    xs = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int32)
    ys = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int32)
    xs = np.clip(xs, 0, image_w - 1)
    ys = np.clip(ys, 0, image_h - 1)

    sampled_depth = depth_image[ys, xs]
    valid_mask = (
        (sampled_depth >= runtime_params["min_depth_mm"])
        & (sampled_depth <= runtime_params["max_depth_mm"])
    )
    valid_ratio = float(np.count_nonzero(valid_mask)) / float(sample_count)
    return valid_ratio >= runtime_params["min_valid_ratio"]


def dominant_depth_from_mask(depth_image, mask, runtime_params):
    masked_pixels = depth_image[mask > 0]
    if masked_pixels.size == 0:
        return None

    valid_depth = masked_pixels[
        (masked_pixels >= runtime_params["min_depth_mm"])
        & (masked_pixels <= runtime_params["max_depth_mm"])
    ]
    if valid_depth.size == 0:
        return None

    valid_ratio = valid_depth.size / masked_pixels.size

    bins = np.arange(
        runtime_params["min_depth_mm"],
        runtime_params["max_depth_mm"] + runtime_params["depth_bin_mm"],
        runtime_params["depth_bin_mm"],
    )
    hist, edges = np.histogram(valid_depth, bins=bins)

    if hist.size == 0:
        return None

    best_bin_index = int(np.argmax(hist))
    if hist[best_bin_index] <= 0:
        return None

    low_edge = edges[best_bin_index]
    high_edge = edges[best_bin_index + 1]

    winning_depths = valid_depth[
        (valid_depth >= low_edge) & (valid_depth < high_edge)
    ]
    if winning_depths.size == 0:
        return None

    majority_ratio = winning_depths.size / valid_depth.size
    dominant_depth_m = float(np.median(winning_depths)) / 1000.0
    dominant_depth_mm = float(np.median(winning_depths))
    coherence_tolerance_mm = max(
        runtime_params["depth_bin_mm"] * BACKGROUND_DEPTH_TOLERANCE_BINS,
        runtime_params["depth_bin_mm"],
    )
    coherent_pixels = masked_pixels[
        (masked_pixels >= runtime_params["min_depth_mm"])
        & (masked_pixels <= runtime_params["max_depth_mm"])
        & (np.abs(masked_pixels.astype(np.float32) - dominant_depth_mm) <= coherence_tolerance_mm)
    ]
    background_ratio = 1.0 - (coherent_pixels.size / masked_pixels.size)

    return {
        "depth_m": dominant_depth_m,
        "valid_ratio": float(valid_ratio),
        "majority_ratio": float(majority_ratio),
        "background_ratio": float(background_ratio),
    }


def branch_color(horizontal_dev_deg, depth_m, runtime_params):
    if horizontal_dev_deg > YELLOW_HORIZONTAL_DEV_DEG:
        return BOX_RED
    if depth_m <= runtime_params["green_threshold_m"]:
        return BOX_GREEN
    return BOX_YELLOW


def candidate_score(candidate, image_width):
    center_x = candidate["center"][0]
    center_bonus = 1.0 - abs(center_x - image_width / 2.0) / (image_width / 2.0)
    center_bonus = np.clip(center_bonus, 0.0, 1.0)

    horizontal_bonus = 1.0 - candidate["horizontal_dev_deg"] / MAX_HORIZONTAL_DEVIATION_DEG
    horizontal_bonus = np.clip(horizontal_bonus, 0.0, 1.0)

    length_bonus = np.clip(candidate["estimated_length_m"] / 0.40, 0.0, 1.0)

    return (
        0.35 * center_bonus
        + 0.25 * horizontal_bonus
        + 0.25 * length_bonus
        + 0.15 * candidate["majority_ratio"]
    )


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def suppress_overlapping_candidates(candidates, iou_threshold=0.45):
    kept = []

    for candidate in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if any(bbox_iou(candidate["bbox"], existing["bbox"]) > iou_threshold for existing in kept):
            continue
        kept.append(candidate)

    return kept


def blend_debug_image(previous_image, current_image, alpha):
    if previous_image is None or previous_image.shape != current_image.shape:
        return current_image.copy()
    return cv2.addWeighted(current_image, alpha, previous_image, 1.0 - alpha, 0.0)


def clamp_box_points(box_points, width, height):
    clamped = box_points.astype(np.float32).copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, height - 1)
    return np.rint(clamped).astype(np.int32)


def smooth_candidate(previous_candidate, current_candidate, image_w, image_h):
    alpha = TRACK_SMOOTHING_ALPHA
    smoothed = dict(current_candidate)

    previous_box = previous_candidate["box"].astype(np.float32)
    current_box = current_candidate["box"].astype(np.float32)
    blended_box = (1.0 - alpha) * previous_box + alpha * current_box
    smoothed["box"] = clamp_box_points(blended_box, image_w, image_h)
    box_x, box_y, box_w, box_h = cv2.boundingRect(smoothed["box"])
    smoothed["bbox"] = clamp_bbox(
        (box_x, box_y, box_x + box_w, box_y + box_h), image_w, image_h,
    )

    previous_center = np.array(previous_candidate["center"], dtype=np.float32)
    current_center = np.array(current_candidate["center"], dtype=np.float32)
    blended_center = (1.0 - alpha) * previous_center + alpha * current_center
    smoothed["center"] = (float(blended_center[0]), float(blended_center[1]))

    for key in (
        "major_axis", "minor_axis", "horizontal_dev_deg", "depth_m",
        "line_a_length_m", "line_b_length_m", "estimated_length_m",
        "valid_ratio", "majority_ratio", "background_ratio", "score",
    ):
        smoothed[key] = float(
            (1.0 - alpha) * float(previous_candidate[key])
            + alpha * float(current_candidate[key])
        )

    if (
        previous_candidate.get("pair_gap_m") is not None
        and current_candidate.get("pair_gap_m") is not None
    ):
        smoothed["pair_gap_m"] = float(
            (1.0 - alpha) * float(previous_candidate["pair_gap_m"])
            + alpha * float(current_candidate["pair_gap_m"])
        )
    else:
        smoothed["pair_gap_m"] = current_candidate.get("pair_gap_m")

    return smoothed


def match_tracked_candidate(tracked_candidate, candidates):
    if tracked_candidate is None or not candidates:
        return None

    tracked_center = np.array(tracked_candidate["center"], dtype=np.float32)
    best_match = None
    best_score = None

    for candidate in candidates:
        candidate_center = np.array(candidate["center"], dtype=np.float32)
        center_distance = float(np.linalg.norm(candidate_center - tracked_center))
        overlap = bbox_iou(tracked_candidate["bbox"], candidate["bbox"])

        if center_distance > TRACK_MATCH_MAX_CENTER_PX and overlap < TRACK_MATCH_MIN_IOU:
            continue

        match_score = overlap + 0.25 * candidate["score"] - 0.002 * center_distance
        if best_score is None or match_score > best_score:
            best_score = match_score
            best_match = candidate

    return best_match


def detect_branch_candidates(color_image, depth_image, focal_length_px, runtime_params, prior_lines=None):
    image_h, image_w = color_image.shape[:2]
    top_limit = int(image_h * TOP_REGION_FRACTION)

    search_region = color_image[:top_limit, :]
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=runtime_params["clahe_clip_limit"],
        tileGridSize=CLAHE_TILE_GRID,
    )
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

    edges = cv2.Canny(blurred, runtime_params["canny_low"], runtime_params["canny_high"])

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MASK_ERODE_KERNEL)

    candidates = []
    reject_reasons = {
        "no_lines": 0, "angle": 0, "duplicates": 0, "pair_angle": 0,
        "pair_gap": 0, "background_hough": 0,
        "minor_axis": 0, "length": 0, "background": 0,
        "depth_none": 0, "valid_ratio": 0, "majority_ratio": 0,
        "depth_range": 0,
    }

    raw_hough_lines = []
    line_gap_schedule = build_hough_gap_schedule(focal_length_px, runtime_params)
    min_line_length_schedule = build_hough_min_length_schedule(focal_length_px, runtime_params)
    for line_gap_px in line_gap_schedule:
        for min_line_length_px in min_line_length_schedule:
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180.0,
                threshold=runtime_params["hough_threshold"],
                minLineLength=min_line_length_px,
                maxLineGap=line_gap_px,
            )
            if lines is None:
                continue
            raw_hough_lines.extend(tuple(map(int, line[0])) for line in lines)

    raw_hough_lines = list(dict.fromkeys(raw_hough_lines))
    raw_hough_line_count = len(raw_hough_lines)

    display_hough_lines = list(raw_hough_lines)
    if runtime_params.get("exclude_background_hough_lines", False):
        display_hough_lines = []
        for line in raw_hough_lines:
            if line_has_in_range_depth(depth_image, line, runtime_params):
                display_hough_lines.append(line)
            else:
                reject_reasons["background_hough"] += 1

    current_frame_display_lines = list(display_hough_lines)

    if prior_lines:
        existing = set(display_hough_lines)
        for line in prior_lines:
            if line not in existing:
                display_hough_lines.append(line)

    if not display_hough_lines:
        reject_reasons["no_lines"] += 1
        return candidates, current_frame_display_lines, raw_hough_line_count, reject_reasons, edges

    horizontal_lines = []
    for x1, y1, x2, y2 in display_hough_lines:
        angle = line_angle_degrees(x1, y1, x2, y2)
        horizontal_dev = horizontal_deviation_degrees(angle)
        if horizontal_dev > MAX_HORIZONTAL_DEVIATION_DEG:
            reject_reasons["angle"] += 1
            continue
        horizontal_lines.append((x1, y1, x2, y2))

    filtered_lines = merge_similar_hough_lines(horizontal_lines)
    reject_reasons["duplicates"] += max(0, len(horizontal_lines) - len(filtered_lines))

    for i in range(len(filtered_lines)):
        line_a = filtered_lines[i]
        ax1, ay1, ax2, ay2 = line_a["segment"]

        for j in range(i + 1, len(filtered_lines)):
            line_b = filtered_lines[j]
            bx1, by1, bx2, by2 = line_b["segment"]

            angle_diff = angle_difference_degrees(line_a["angle_deg"], line_b["angle_deg"])
            if angle_diff > MAX_PAIR_ANGLE_DIFF_DEG:
                reject_reasons["pair_angle"] += 1
                continue

            pair_points = np.array(
                [(ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2)], dtype=np.int32,
            )
            pair_hull = cv2.convexHull(pair_points)
            rect = cv2.minAreaRect(pair_hull)
            (cx, cy), (w, h), _ = rect
            major_axis = max(w, h)
            minor_axis = min(w, h)

            if minor_axis <= 0:
                reject_reasons["minor_axis"] += 1
                continue

            box = cv2.boxPoints(rect)
            box = np.int32(box)

            horizontal_angle = longest_edge_angle_deg(box)
            horizontal_dev = horizontal_deviation_degrees(horizontal_angle)
            if horizontal_dev > MAX_HORIZONTAL_DEVIATION_DEG:
                reject_reasons["angle"] += 1
                continue

            mask = np.zeros((top_limit, image_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pair_hull.reshape(-1, 2), 255)
            mask = cv2.erode(mask, erode_kernel, iterations=1)

            full_mask = np.zeros(depth_image.shape, dtype=np.uint8)
            full_mask[:top_limit, :] = mask

            depth_vote = dominant_depth_from_mask(depth_image, full_mask, runtime_params)
            if depth_vote is None:
                reject_reasons["depth_none"] += 1
                continue
            if depth_vote["background_ratio"] > runtime_params["max_background_ratio"]:
                reject_reasons["background"] += 1
                continue
            if depth_vote["valid_ratio"] < runtime_params["min_valid_ratio"]:
                reject_reasons["valid_ratio"] += 1
                continue
            if depth_vote["majority_ratio"] < runtime_params["min_majority_ratio"]:
                reject_reasons["majority_ratio"] += 1
                continue

            depth_m = depth_vote["depth_m"]
            if not (runtime_params["min_depth_m"] <= depth_m <= runtime_params["max_depth_m"]):
                reject_reasons["depth_range"] += 1
                continue

            line_a_length_m = estimate_length_m(line_a["length_px"], depth_m, focal_length_px)
            line_b_length_m = estimate_length_m(line_b["length_px"], depth_m, focal_length_px)
            if line_a_length_m is None or line_b_length_m is None:
                reject_reasons["length"] += 1
                continue
            if min(line_a_length_m, line_b_length_m) < MIN_ESTIMATED_LENGTH_M:
                reject_reasons["length"] += 1
                continue

            pair_gap_px = line_pair_gap_px(line_a["segment"], line_b["segment"])
            estimated_gap_m = None
            if pair_gap_px >= MIN_MEASURABLE_PAIR_GAP_PX:
                estimated_gap_m = estimate_length_m(pair_gap_px, depth_m, focal_length_px)
                if estimated_gap_m is None:
                    reject_reasons["pair_gap"] += 1
                    continue
                if not (runtime_params["pair_min_gap_m"] <= estimated_gap_m <= runtime_params["pair_max_gap_m"]):
                    reject_reasons["pair_gap"] += 1
                    continue

            estimated_length_m = 0.5 * (line_a_length_m + line_b_length_m)

            bbox = clamp_bbox(
                (
                    min(ax1, ax2, bx1, bx2) - PAIR_PAD_X,
                    min(ay1, ay2, by1, by2) - PAIR_PAD_Y,
                    max(ax1, ax2, bx1, bx2) + PAIR_PAD_X,
                    max(ay1, ay2, by1, by2) + PAIR_PAD_Y,
                ),
                image_w, image_h,
            )

            candidate = {
                "lines": (line_a["segment"], line_b["segment"]),
                "box": box, "bbox": bbox,
                "center": (float(cx), float(cy)),
                "major_axis": float(major_axis),
                "minor_axis": float(minor_axis),
                "horizontal_dev_deg": float(horizontal_dev),
                "depth_m": depth_m,
                "line_a_length_m": float(line_a_length_m),
                "line_b_length_m": float(line_b_length_m),
                "pair_gap_m": (float(estimated_gap_m) if estimated_gap_m is not None else None),
                "estimated_length_m": float(estimated_length_m),
                "valid_ratio": depth_vote["valid_ratio"],
                "majority_ratio": depth_vote["majority_ratio"],
                "background_ratio": depth_vote["background_ratio"],
                "mask": full_mask,
            }
            candidate["score"] = candidate_score(candidate, image_w)
            candidates.append(candidate)

    candidates = suppress_overlapping_candidates(candidates)
    return candidates, current_frame_display_lines, raw_hough_line_count, reject_reasons, edges


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
pipeline, profile, active_profile = start_pipeline_with_fallback()
align = rs.align(rs.stream.color)

COLOR_WIDTH = active_profile["color_width"]
COLOR_HEIGHT = active_profile["color_height"]
color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_intrinsics = color_stream_profile.get_intrinsics()
COLOR_FOCAL_LENGTH_PX = float(color_intrinsics.fx)
print(f"Color focal length: {COLOR_FOCAL_LENGTH_PX:.2f}px")

depth_sensor = profile.get_device().first_depth_sensor()

try:
    depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
except Exception:
    pass
try:
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
except Exception:
    pass
try:
    depth_sensor.set_option(rs.option.laser_power, 180)
except Exception:
    pass

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.35)
spatial.set_option(rs.option.filter_smooth_delta, 30)

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.25)
temporal.set_option(rs.option.filter_smooth_delta, 30)

if HEADLESS:
    gui = None
else:
    gui = ParamGUI("params.yaml")
last_camera_settings = {}


def _load_headless_params():
    from pathlib import Path
    defaults = {
        "canny": {"canny_low": 19, "canny_high": 69, "clahe_clip_limit": 6.2},
        "hough": {
            "hough_threshold": 131, "hough_min_line_length": 142,
            "gap_ref_px": 28, "gap_min_px": 105, "gap_max_px": 206,
            "pair_min_gap_inches_x10": 10, "pair_max_gap_inches_x10": 35,
            "exclude_background_hough_lines": 1,
        },
        "depth": {
            "min_depth_cm": 4, "max_depth_cm": 200, "green_threshold_cm": 20,
            "depth_bin_mm": 20, "min_valid_percent": 5,
            "min_majority_percent": 18, "max_background_percent": 60,
        },
        "camera": {
            "visual_preset": 4, "emitter_enabled": 1, "laser_power": 180,
            "auto_exposure": 1, "exposure": 8500, "gain": 16,
            "spatial_magnitude": 1, "spatial_alpha_x100": 35, "spatial_delta": 30,
            "temporal_alpha_x100": 25, "temporal_delta": 30,
        },
    }
    p = defaults
    yp = Path("params.yaml")
    if yp.exists():
        with open(yp) as f:
            on_disk = yaml.safe_load(f) or {}
        for section in p:
            if section in on_disk and isinstance(on_disk[section], dict):
                p[section].update(on_disk[section])
    c, h, d, cam = p["canny"], p["hough"], p["depth"], p["camera"]
    canny_low = int(max(0, min(254, int(c["canny_low"]))))
    canny_high = int(max(canny_low + 1, min(255, int(c["canny_high"]))))
    gap_min_px = max(1, int(h["gap_min_px"]))
    min_depth_m = max(0.01, int(d["min_depth_cm"]) / 100.0)
    max_depth_m = max(min_depth_m + 0.01, int(d["max_depth_cm"]) / 100.0)
    green_threshold_m = min(max(int(d["green_threshold_cm"]) / 100.0, min_depth_m), max_depth_m)
    return {
        "canny_low": canny_low, "canny_high": canny_high,
        "clahe_clip_limit": max(0.1, float(c["clahe_clip_limit"])),
        "hough_threshold": max(1, int(h["hough_threshold"])),
        "hough_min_line_length": max(1, int(h["hough_min_line_length"])),
        "gap_ref_px": max(1, int(h["gap_ref_px"])),
        "gap_min_px": gap_min_px, "gap_max_px": max(gap_min_px, int(h["gap_max_px"])),
        "pair_min_gap_m": max(0.0, int(h["pair_min_gap_inches_x10"]) / 10.0 / 39.3701),
        "pair_max_gap_m": max(0.0, int(h["pair_max_gap_inches_x10"]) / 10.0 / 39.3701),
        "exclude_background_hough_lines": bool(int(h["exclude_background_hough_lines"])),
        "min_depth_m": float(min_depth_m), "max_depth_m": float(max_depth_m),
        "green_threshold_m": float(green_threshold_m),
        "min_depth_mm": int(round(min_depth_m * 1000.0)),
        "max_depth_mm": int(round(max_depth_m * 1000.0)),
        "depth_bin_mm": max(1, int(d["depth_bin_mm"])),
        "min_valid_ratio": min(1.0, max(0.0, int(d["min_valid_percent"]) / 100.0)),
        "min_majority_ratio": min(1.0, max(0.0, int(d["min_majority_percent"]) / 100.0)),
        "max_background_ratio": min(1.0, max(0.0, int(d["max_background_percent"]) / 100.0)),
        "visual_preset": int(cam["visual_preset"]), "emitter_enabled": int(cam["emitter_enabled"]),
        "laser_power": int(cam["laser_power"]), "auto_exposure": int(cam["auto_exposure"]),
        "exposure": max(1, int(cam["exposure"])), "gain": max(1, int(cam["gain"])),
        "spatial_magnitude": max(1, int(cam["spatial_magnitude"])),
        "spatial_alpha": max(0.01, int(cam["spatial_alpha_x100"]) / 100.0),
        "spatial_delta": max(1, int(cam["spatial_delta"])),
        "temporal_alpha": max(0.01, int(cam["temporal_alpha_x100"]) / 100.0),
        "temporal_delta": max(1, int(cam["temporal_delta"])),
    }


_headless_params = None

STM32_PORT = '/dev/ttyACM0'
STM32_BAUD = 115200
GREEN_HOLD_SECONDS = 0.1

try:
    stm32_ser = serial.Serial(STM32_PORT, STM32_BAUD, timeout=1)
    print(f"Connected to STM32 on {STM32_PORT}")
    time.sleep(2)
    stm32_ser.write(b'curl 1500 3\n')
    print("Sent to STM32: curl 1500 3 (tuck in)")
except serial.SerialException as e:
    print(f"Warning: Could not open STM32 serial port: {e}")
    stm32_ser = None

green_start_time = None
curl_sent = False
tracked_best_candidate = None
tracked_missing_frames = 0
previous_raw_hough_image = None
previous_canny_debug_image = None
hough_line_history = []
compute_line_history = []

_quit_flag = threading.Event()


def _stdin_listener():
    try:
        while not _quit_flag.is_set():
            if select.select([sys.stdin], [], [], 0.5)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() == "q":
                    print("\n*** 'q' received — shutting down ***")
                    _quit_flag.set()
                    return
                if line and stm32_ser is not None and stm32_ser.is_open:
                    stm32_ser.write((line + '\n').encode())
                    print(f"Sent to STM32: {line}")
                elif line:
                    print("STM32 not connected — command ignored")
    except Exception:
        pass


_stdin_thread = threading.Thread(target=_stdin_listener, daemon=True)
_stdin_thread.start()
print("Commands: type 'q' to quit, or any STM32 command (curl, uncurl, motor_stop, motor_status, etc.)")

try:
    while not _quit_flag.is_set():
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        if not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        if not HEADLESS:
            display_image = color_image.copy()
        depth_raw = np.asanyarray(depth_frame.get_data())

        if depth_raw.shape[1] != COLOR_WIDTH or depth_raw.shape[0] != COLOR_HEIGHT:
            depth_image = cv2.resize(depth_raw, (COLOR_WIDTH, COLOR_HEIGHT), interpolation=cv2.INTER_NEAREST)
        else:
            depth_image = depth_raw

        if HEADLESS:
            if _headless_params is None:
                _headless_params = _load_headless_params()
            runtime_params = _headless_params
        else:
            gui.update()
            runtime_params = gui.get_all_params()

        apply_camera_runtime_params(depth_sensor, spatial, temporal, runtime_params, last_camera_settings)

        if not HEADLESS:
            clipped = np.clip(depth_image, runtime_params["min_depth_mm"], runtime_params["max_depth_mm"])
            normalized = ((clipped - runtime_params["min_depth_mm"]) / max(1, runtime_params["max_depth_mm"] - runtime_params["min_depth_mm"]) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            top_limit = int(COLOR_HEIGHT * TOP_REGION_FRACTION)
            raw_hough_image = np.zeros((COLOR_HEIGHT, COLOR_WIDTH, 3), dtype=np.uint8)
            if DRAW_TOP_REGION_LINE:
                cv2.line(display_image, (0, top_limit), (COLOR_WIDTH, top_limit), TOP_REGION_COLOR, 2)
                cv2.line(depth_colormap, (0, top_limit), (COLOR_WIDTH, top_limit), (255, 255, 255), 2)

        compute_prune_time = time.time()
        compute_line_history = [
            (t, l) for t, l in compute_line_history
            if compute_prune_time - t <= HOUGH_LINE_COMPUTE_LINGER_SECONDS
        ]
        prior_lines = list(dict.fromkeys(l for t, l in compute_line_history))

        candidates, raw_hough_lines, raw_hough_line_count, reject_reasons, canny_edges = detect_branch_candidates(
            color_image, depth_image, COLOR_FOCAL_LENGTH_PX, runtime_params, prior_lines=prior_lines,
        )

        for line in raw_hough_lines:
            compute_line_history.append((time.time(), line))

        active_rejects = {k: v for k, v in reject_reasons.items() if v > 0}
        if active_rejects:
            print(f"Rejects: {active_rejects}  |  Accepted: {len(candidates)}")

        best_candidate = max(candidates, key=lambda c: c["score"]) if candidates else None

        matched_candidate = match_tracked_candidate(tracked_best_candidate, candidates)
        if matched_candidate is not None and tracked_best_candidate is not None:
            tracked_best_candidate = smooth_candidate(tracked_best_candidate, matched_candidate, COLOR_WIDTH, COLOR_HEIGHT)
            tracked_missing_frames = 0
        elif best_candidate is not None and tracked_best_candidate is None:
            tracked_best_candidate = dict(best_candidate)
            tracked_missing_frames = 0
        elif best_candidate is not None and tracked_missing_frames > TRACK_MAX_MISSING_FRAMES:
            tracked_best_candidate = dict(best_candidate)
            tracked_missing_frames = 0
        elif tracked_best_candidate is not None:
            tracked_missing_frames += 1
            if tracked_missing_frames > TRACK_MAX_MISSING_FRAMES:
                tracked_best_candidate = None
                tracked_missing_frames = 0

        if not HEADLESS:
            canny_debug_base = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
            for line in raw_hough_lines:
                hough_line_history.append((time.time(), line))
            current_time = time.time()
            hough_line_history = [
                (timestamp, line) for timestamp, line in hough_line_history
                if current_time - timestamp <= HOUGH_LINE_LINGER_SECONDS
            ]
            for timestamp, line in hough_line_history:
                x1, y1, x2, y2 = line
                age_ratio = np.clip((current_time - timestamp) / HOUGH_LINE_LINGER_SECONDS, 0.0, 1.0)
                intensity = int(round(255 * (1.0 - 0.7 * age_ratio)))
                cv2.line(raw_hough_image, (x1, y1), (x2, y2), (0, intensity, intensity), 2)
            raw_hough_image = blend_debug_image(previous_raw_hough_image, raw_hough_image, DEBUG_IMAGE_ALPHA)
            canny_debug_image = blend_debug_image(previous_canny_debug_image, canny_debug_base, DEBUG_IMAGE_ALPHA)
            previous_raw_hough_image = raw_hough_image.copy()
            previous_canny_debug_image = canny_debug_image.copy()
            for candidate in candidates:
                color = branch_color(candidate["horizontal_dev_deg"], candidate["depth_m"], runtime_params)
                cv2.polylines(display_image, [candidate["box"]], True, color, 2)
                cx, cy = candidate["center"]
                cv2.circle(display_image, (int(cx), int(cy)), 4, color, -1)
                x, y, w, h = cv2.boundingRect(candidate["box"])
                cv2.putText(display_image, f"{candidate['depth_m']:.2f}m", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            display_best_candidate = tracked_best_candidate
            if display_best_candidate is not None:
                best_color = branch_color(display_best_candidate["horizontal_dev_deg"], display_best_candidate["depth_m"], runtime_params)
                cv2.polylines(display_image, [display_best_candidate["box"]], True, best_color, 4)
                best_cx, best_cy = display_best_candidate["center"]
                cv2.circle(display_image, (int(best_cx), int(best_cy)), 5, best_color, -1)

        green_candidates = [
            c for c in candidates
            if c["horizontal_dev_deg"] <= YELLOW_HORIZONTAL_DEV_DEG
            and c["depth_m"] <= runtime_params["green_threshold_m"]
        ]
        yellow_candidates = [
            c for c in candidates
            if c["horizontal_dev_deg"] <= YELLOW_HORIZONTAL_DEV_DEG
            and c["depth_m"] > runtime_params["green_threshold_m"]
        ]
        red_candidates = [
            c for c in candidates
            if c["horizontal_dev_deg"] > YELLOW_HORIZONTAL_DEV_DEG
        ]

        if len(green_candidates) > 0:
            if green_start_time is None:
                green_start_time = time.time()
            elif not curl_sent and (time.time() - green_start_time) >= GREEN_HOLD_SECONDS:
                if stm32_ser is not None and stm32_ser.is_open:
                    stm32_ser.write(b'curl -1750 3 1600 4\n')
                    print("Sent to STM32: curl -1750 3 1600 4")
                curl_sent = True
        else:
            green_start_time = None

        if not HEADLESS:
            cv2.putText(display_image, f"Candidates: {len(red_candidates)}R {len(yellow_candidates)}Y {len(green_candidates)}G", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Close + horizontal (green): {len(green_candidates)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BOX_GREEN, 2)
            if yellow_candidates:
                best_yellow = min(yellow_candidates, key=lambda c: c["horizontal_dev_deg"])
                yellow_text = f"Best yellow: {best_yellow['horizontal_dev_deg']:.1f} deg, {best_yellow['depth_m']:.2f}m"
            else:
                yellow_text = "Yellow: none"
            cv2.putText(display_image, yellow_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.60, BOX_YELLOW, 2)
            cv2.putText(raw_hough_image, (f"Hough lines: {len(raw_hough_lines)}/{raw_hough_line_count}" if runtime_params.get("exclude_background_hough_lines", False) else f"Hough lines: {len(raw_hough_lines)}"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(raw_hough_image, f"Visual {int(round(HOUGH_LINE_LINGER_SECONDS * 1000.0))} ms  Compute {int(round(HOUGH_LINE_COMPUTE_LINGER_SECONDS * 1000.0))} ms", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            display_best_candidate = tracked_best_candidate
            if display_best_candidate is not None:
                best_c = branch_color(display_best_candidate["horizontal_dev_deg"], display_best_candidate["depth_m"], runtime_params)
                cv2.putText(display_image, f"Best depth: {display_best_candidate['depth_m']:.2f} m", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, best_c, 2)
                cv2.putText(display_image, f"Length: {display_best_candidate['estimated_length_m'] * 39.3701:.1f} in", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                width_text = "Width: n/a"
                if display_best_candidate.get("pair_gap_m") is not None:
                    width_text = f"Width: {display_best_candidate['pair_gap_m'] * 39.3701:.1f} in"
                cv2.putText(display_image, width_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                stability_text = "Tracking: stable" if tracked_missing_frames == 0 else f"Tracking hold: {tracked_missing_frames}"
                cv2.putText(display_image, stability_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Depth range: {runtime_params['min_depth_m']:.2f}m to {runtime_params['max_depth_m']:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Green {runtime_params['green_threshold_m']:.2f}m  Bin {runtime_params['depth_bin_mm']}mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Valid {runtime_params['min_valid_ratio']:.2f}  Majority {runtime_params['min_majority_ratio']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"Max background {runtime_params['max_background_ratio']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(raw_hough_image, "Raw Hough Lines", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(raw_hough_image, f"Hough {runtime_params['hough_threshold']}  MinLen {runtime_params['hough_min_line_length']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(raw_hough_image, f"StitchGap {runtime_params['gap_min_px']}-{runtime_params['gap_max_px']} px", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(raw_hough_image, f"PairGap {runtime_params['pair_min_gap_m'] * 39.3701:.1f}-{runtime_params['pair_max_gap_m'] * 39.3701:.1f} in", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(canny_debug_image, "Canny Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(canny_debug_image, f"Canny {runtime_params['canny_low']}/{runtime_params['canny_high']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(canny_debug_image, f"CLAHE x10 {int(round(runtime_params['clahe_clip_limit'] * 10.0))}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(raw_hough_image, ("Exclude BG Hough: on" if runtime_params.get("exclude_background_hough_lines", False) else "Exclude BG Hough: off"), (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            combined = np.vstack((np.hstack((display_image, depth_colormap)), np.hstack((raw_hough_image, canny_debug_image))))
            cv2.imshow(MAIN_WINDOW_NAME, combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _quit_flag.set()

finally:
    pipeline.stop()
    if not HEADLESS:
        gui.root.destroy()
        cv2.destroyAllWindows()
    if stm32_ser is not None and stm32_ser.is_open:
        stm32_ser.close()
        print("STM32 serial connection closed")
