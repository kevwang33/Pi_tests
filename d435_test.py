import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise ImportError(
        "This script requires `pyrealsense2` for the Intel RealSense D435 camera. "
        "Install dependencies with: pip install numpy opencv-python pyrealsense2"
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
TOP_REGION_FRACTION = 1.0 / 2.0

# -----------------------------
# RGB preprocessing
# -----------------------------
CANNY_LOW = 19
CANNY_HIGH = 69
GAUSSIAN_KERNEL = (5, 5)

CLAHE_CLIP_LIMIT = 6.2
CLAHE_TILE_GRID = (8, 8)

# Tighter morphology than before to avoid grabbing too much nearby clutter.
MORPH_CLOSE_KERNEL = (13, 5)
MORPH_OPEN_KERNEL = (5, 3)

# Used to tighten the filled line-pair mask before depth voting.
MASK_ERODE_KERNEL = (5, 3)

# -----------------------------
# Hough-line / line-pair filters
# -----------------------------
HOUGH_THRESHOLD = 131
HOUGH_MIN_LINE_LENGTH = 142
HOUGH_LINE_CONNECT_GAP_REFERENCE_PX = 28
HOUGH_LINE_CONNECT_GAP_REFERENCE_DEPTH_M = 1.0
HOUGH_LINE_CONNECT_GAP_MIN_PX = 105
HOUGH_LINE_CONNECT_GAP_MAX_PX = 206

MAX_HORIZONTAL_DEVIATION_DEG = 15.0
MAX_PAIR_ANGLE_DIFF_DEG = 12.0
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

# -----------------------------
# Display / debugging
# -----------------------------
DRAW_TOP_REGION_LINE = True
TOP_REGION_COLOR = (255, 255, 0)
BOX_RED = (0, 0, 255)
BOX_GREEN = (0, 255, 0)
MAIN_WINDOW_NAME = "D435 Hough Line Branch Detection"
CANNY_CONTROL_WINDOW_NAME = "Canny Controls"
CONTROL_WINDOW_NAME = "Raw Hough Controls"
DEPTH_CONTROL_WINDOW_NAME = "Depth Controls"
CAMERA_CONTROL_WINDOW_NAME = "Camera Controls"


def _noop(_value):
    pass


def render_label_panel(title, rows, width=520, row_height=34):
    height = 50 + row_height * len(rows)
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        panel,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    for index, (label, value) in enumerate(rows):
        y = 55 + index * row_height
        cv2.putText(
            panel,
            f"{label}: {value}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
        )

    return panel


def format_slider_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_control_window(window_name, title, rows, width=520, row_height=34):
    panel = render_label_panel(
        title,
        [(label, format_slider_value(value)) for label, value in rows],
        width=width,
        row_height=row_height,
    )
    cv2.imshow(window_name, panel)


def create_canny_trackbars():
    cv2.namedWindow(CANNY_CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CANNY_CONTROL_WINDOW_NAME, 520, 220)
    cv2.createTrackbar("Canny Low", CANNY_CONTROL_WINDOW_NAME, CANNY_LOW, 255, _noop)
    cv2.createTrackbar("Canny High", CANNY_CONTROL_WINDOW_NAME, CANNY_HIGH, 255, _noop)
    cv2.createTrackbar(
        "CLAHE x10",
        CANNY_CONTROL_WINDOW_NAME,
        int(round(CLAHE_CLIP_LIMIT * 10.0)),
        100,
        _noop,
    )


def get_canny_runtime_params():
    canny_low = cv2.getTrackbarPos("Canny Low", CANNY_CONTROL_WINDOW_NAME)
    canny_high = cv2.getTrackbarPos("Canny High", CANNY_CONTROL_WINDOW_NAME)
    clahe_clip_x10 = cv2.getTrackbarPos("CLAHE x10", CANNY_CONTROL_WINDOW_NAME)

    canny_low = int(np.clip(canny_low, 0, 254))
    canny_high = int(np.clip(max(canny_high, canny_low + 1), 1, 255))
    clahe_clip_limit = max(0.1, clahe_clip_x10 / 10.0)

    return {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "clahe_clip_limit": clahe_clip_limit,
    }


def create_raw_hough_trackbars():
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, 520, 240)
    cv2.createTrackbar("Hough Thresh", CONTROL_WINDOW_NAME, HOUGH_THRESHOLD, 300, _noop)
    cv2.createTrackbar(
        "Min Line Len", CONTROL_WINDOW_NAME, HOUGH_MIN_LINE_LENGTH, 400, _noop
    )
    cv2.createTrackbar(
        "Gap Ref Px",
        CONTROL_WINDOW_NAME,
        HOUGH_LINE_CONNECT_GAP_REFERENCE_PX,
        200,
        _noop,
    )
    cv2.createTrackbar(
        "Gap Min Px",
        CONTROL_WINDOW_NAME,
        HOUGH_LINE_CONNECT_GAP_MIN_PX,
        200,
        _noop,
    )
    cv2.createTrackbar(
        "Gap Max Px",
        CONTROL_WINDOW_NAME,
        HOUGH_LINE_CONNECT_GAP_MAX_PX,
        300,
        _noop,
    )


def get_raw_hough_runtime_params():
    hough_threshold = cv2.getTrackbarPos("Hough Thresh", CONTROL_WINDOW_NAME)
    hough_min_line_length = cv2.getTrackbarPos("Min Line Len", CONTROL_WINDOW_NAME)
    gap_ref_px = cv2.getTrackbarPos("Gap Ref Px", CONTROL_WINDOW_NAME)
    gap_min_px = cv2.getTrackbarPos("Gap Min Px", CONTROL_WINDOW_NAME)
    gap_max_px = cv2.getTrackbarPos("Gap Max Px", CONTROL_WINDOW_NAME)

    hough_threshold = max(1, hough_threshold)
    hough_min_line_length = max(1, hough_min_line_length)
    gap_ref_px = max(1, gap_ref_px)
    gap_min_px = max(1, gap_min_px)
    gap_max_px = max(gap_min_px, gap_max_px)

    return {
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "gap_ref_px": gap_ref_px,
        "gap_min_px": gap_min_px,
        "gap_max_px": gap_max_px,
    }


def create_depth_trackbars():
    cv2.namedWindow(DEPTH_CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEPTH_CONTROL_WINDOW_NAME, 520, 280)
    cv2.createTrackbar(
        "Min Depth cm",
        DEPTH_CONTROL_WINDOW_NAME,
        int(round(MIN_DEPTH_M * 100.0)),
        500,
        _noop,
    )
    cv2.createTrackbar(
        "Max Depth cm",
        DEPTH_CONTROL_WINDOW_NAME,
        int(round(MAX_DEPTH_M * 100.0)),
        500,
        _noop,
    )
    cv2.createTrackbar(
        "Green cm",
        DEPTH_CONTROL_WINDOW_NAME,
        int(round(GREEN_THRESHOLD_M * 100.0)),
        500,
        _noop,
    )
    cv2.createTrackbar(
        "Depth Bin mm",
        DEPTH_CONTROL_WINDOW_NAME,
        DEPTH_BIN_MM,
        100,
        _noop,
    )
    cv2.createTrackbar(
        "Min Valid %",
        DEPTH_CONTROL_WINDOW_NAME,
        int(round(MIN_VALID_DEPTH_RATIO * 100.0)),
        100,
        _noop,
    )
    cv2.createTrackbar(
        "Min Majority %",
        DEPTH_CONTROL_WINDOW_NAME,
        int(round(MIN_MAJORITY_RATIO * 100.0)),
        100,
        _noop,
    )


def get_depth_runtime_params():
    min_depth_cm = cv2.getTrackbarPos("Min Depth cm", DEPTH_CONTROL_WINDOW_NAME)
    max_depth_cm = cv2.getTrackbarPos("Max Depth cm", DEPTH_CONTROL_WINDOW_NAME)
    green_cm = cv2.getTrackbarPos("Green cm", DEPTH_CONTROL_WINDOW_NAME)
    depth_bin_mm = cv2.getTrackbarPos("Depth Bin mm", DEPTH_CONTROL_WINDOW_NAME)
    min_valid_percent = cv2.getTrackbarPos("Min Valid %", DEPTH_CONTROL_WINDOW_NAME)
    min_majority_percent = cv2.getTrackbarPos("Min Majority %", DEPTH_CONTROL_WINDOW_NAME)

    min_depth_m = max(0.01, min_depth_cm / 100.0)
    max_depth_m = max(min_depth_m + 0.01, max_depth_cm / 100.0)
    green_threshold_m = np.clip(green_cm / 100.0, min_depth_m, max_depth_m)
    depth_bin_mm = max(1, depth_bin_mm)

    return {
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "green_threshold_m": float(green_threshold_m),
        "min_depth_mm": int(round(min_depth_m * 1000.0)),
        "max_depth_mm": int(round(max_depth_m * 1000.0)),
        "depth_bin_mm": int(depth_bin_mm),
        "min_valid_ratio": float(np.clip(min_valid_percent / 100.0, 0.0, 1.0)),
        "min_majority_ratio": float(np.clip(min_majority_percent / 100.0, 0.0, 1.0)),
    }


def create_camera_trackbars():
    cv2.namedWindow(CAMERA_CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_CONTROL_WINDOW_NAME, 560, 420)
    cv2.createTrackbar("Visual Preset", CAMERA_CONTROL_WINDOW_NAME, 4, 5, _noop)
    cv2.createTrackbar("Emitter", CAMERA_CONTROL_WINDOW_NAME, 1, 1, _noop)
    cv2.createTrackbar("Laser Power", CAMERA_CONTROL_WINDOW_NAME, 180, 360, _noop)
    cv2.createTrackbar("Auto Exposure", CAMERA_CONTROL_WINDOW_NAME, 1, 1, _noop)
    cv2.createTrackbar("Exposure", CAMERA_CONTROL_WINDOW_NAME, 8500, 20000, _noop)
    cv2.createTrackbar("Gain", CAMERA_CONTROL_WINDOW_NAME, 16, 128, _noop)
    cv2.createTrackbar("Spatial Mag", CAMERA_CONTROL_WINDOW_NAME, 1, 5, _noop)
    cv2.createTrackbar("Spatial A x100", CAMERA_CONTROL_WINDOW_NAME, 35, 100, _noop)
    cv2.createTrackbar("Spatial Delta", CAMERA_CONTROL_WINDOW_NAME, 30, 100, _noop)
    cv2.createTrackbar("Temporal A x100", CAMERA_CONTROL_WINDOW_NAME, 25, 100, _noop)
    cv2.createTrackbar("Temporal Delta", CAMERA_CONTROL_WINDOW_NAME, 30, 100, _noop)


def get_camera_runtime_params():
    return {
        "visual_preset": cv2.getTrackbarPos("Visual Preset", CAMERA_CONTROL_WINDOW_NAME),
        "emitter_enabled": cv2.getTrackbarPos("Emitter", CAMERA_CONTROL_WINDOW_NAME),
        "laser_power": cv2.getTrackbarPos("Laser Power", CAMERA_CONTROL_WINDOW_NAME),
        "auto_exposure": cv2.getTrackbarPos("Auto Exposure", CAMERA_CONTROL_WINDOW_NAME),
        "exposure": max(1, cv2.getTrackbarPos("Exposure", CAMERA_CONTROL_WINDOW_NAME)),
        "gain": max(1, cv2.getTrackbarPos("Gain", CAMERA_CONTROL_WINDOW_NAME)),
        "spatial_magnitude": max(
            1, cv2.getTrackbarPos("Spatial Mag", CAMERA_CONTROL_WINDOW_NAME)
        ),
        "spatial_alpha": max(
            0.01, cv2.getTrackbarPos("Spatial A x100", CAMERA_CONTROL_WINDOW_NAME) / 100.0
        ),
        "spatial_delta": max(
            1, cv2.getTrackbarPos("Spatial Delta", CAMERA_CONTROL_WINDOW_NAME)
        ),
        "temporal_alpha": max(
            0.01, cv2.getTrackbarPos("Temporal A x100", CAMERA_CONTROL_WINDOW_NAME) / 100.0
        ),
        "temporal_delta": max(
            1, cv2.getTrackbarPos("Temporal Delta", CAMERA_CONTROL_WINDOW_NAME)
        ),
    }


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
        0.35,
        0.50,
        0.75,
        1.00,
        runtime_params["max_depth_m"],
    ]

    gap_schedule = []
    for depth_m in depth_samples_m:
        depth_m = float(
            np.clip(depth_m, runtime_params["min_depth_m"], runtime_params["max_depth_m"])
        )
        gap_px = estimate_pixels_for_length(reference_gap_m, depth_m, focal_length_px)
        if gap_px is None:
            continue

        gap_px = int(
            round(
                np.clip(
                    gap_px,
                    runtime_params["gap_min_px"],
                    runtime_params["gap_max_px"],
                )
            )
        )
        gap_schedule.append(gap_px)

    gap_schedule.append(runtime_params["gap_ref_px"])
    return sorted(set(gap_schedule))


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

    return {
        "depth_m": dominant_depth_m,
        "valid_ratio": float(valid_ratio),
        "majority_ratio": float(majority_ratio),
    }


def depth_to_branch_color(depth_m, runtime_params):
    if depth_m <= runtime_params["green_threshold_m"]:
        return BOX_GREEN
    return BOX_RED


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


def detect_branch_candidates(color_image, depth_image, focal_length_px, runtime_params):
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

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MASK_ERODE_KERNEL)

    binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    candidates = []
    reject_reasons = {
        "no_lines": 0, "angle": 0, "duplicates": 0, "pair_angle": 0,
        "pair_gap": 0,
        "minor_axis": 0, "length": 0,
        "depth_none": 0, "valid_ratio": 0, "majority_ratio": 0,
        "depth_range": 0,
    }

    raw_hough_lines = []
    for line_gap_px in build_hough_gap_schedule(focal_length_px, runtime_params):
        lines = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 180.0,
            threshold=runtime_params["hough_threshold"],
            minLineLength=runtime_params["hough_min_line_length"],
            maxLineGap=line_gap_px,
        )
        if lines is None:
            continue

        raw_hough_lines.extend(tuple(map(int, line[0])) for line in lines)

    raw_hough_lines = list(dict.fromkeys(raw_hough_lines))
    if not raw_hough_lines:
        reject_reasons["no_lines"] += 1
        return candidates, [], reject_reasons, edges
    horizontal_lines = []
    for x1, y1, x2, y2 in raw_hough_lines:
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
                [(ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2)],
                dtype=np.int32,
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

            line_a_length_m = estimate_length_m(
                line_a["length_px"], depth_m, focal_length_px
            )
            line_b_length_m = estimate_length_m(
                line_b["length_px"], depth_m, focal_length_px
            )
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
                if not (MIN_PAIR_GAP_M <= estimated_gap_m <= MAX_PAIR_GAP_M):
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
                image_w,
                image_h,
            )

            candidate = {
                "lines": (line_a["segment"], line_b["segment"]),
                "box": box,
                "bbox": bbox,
                "center": (float(cx), float(cy)),
                "major_axis": float(major_axis),
                "minor_axis": float(minor_axis),
                "horizontal_dev_deg": float(horizontal_dev),
                "depth_m": depth_m,
                "line_a_length_m": float(line_a_length_m),
                "line_b_length_m": float(line_b_length_m),
                "pair_gap_m": (
                    float(estimated_gap_m) if estimated_gap_m is not None else None
                ),
                "estimated_length_m": float(estimated_length_m),
                "valid_ratio": depth_vote["valid_ratio"],
                "majority_ratio": depth_vote["majority_ratio"],
                "mask": full_mask,
            }
            candidate["score"] = candidate_score(candidate, image_w)
            candidates.append(candidate)

    candidates = suppress_overlapping_candidates(candidates)
    return candidates, raw_hough_lines, reject_reasons, edges


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

create_canny_trackbars()
create_raw_hough_trackbars()
create_depth_trackbars()
create_camera_trackbars()
last_camera_settings = {}
print("Press 'q' to quit.")

try:
    while True:
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
        display_image = color_image.copy()
        depth_raw = np.asanyarray(depth_frame.get_data())

        if depth_raw.shape[1] != COLOR_WIDTH or depth_raw.shape[0] != COLOR_HEIGHT:
            depth_image = cv2.resize(
                depth_raw,
                (COLOR_WIDTH, COLOR_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            depth_image = depth_raw

        canny_params = get_canny_runtime_params()
        hough_params = get_raw_hough_runtime_params()
        depth_params = get_depth_runtime_params()
        camera_params = get_camera_runtime_params()
        runtime_params = {**canny_params, **hough_params, **depth_params, **camera_params}

        apply_camera_runtime_params(
            depth_sensor,
            spatial,
            temporal,
            runtime_params,
            last_camera_settings,
        )

        clipped = np.clip(
            depth_image,
            runtime_params["min_depth_mm"],
            runtime_params["max_depth_mm"],
        )
        normalized = (
            (clipped - runtime_params["min_depth_mm"])
            / max(
                1,
                runtime_params["max_depth_mm"] - runtime_params["min_depth_mm"],
            )
            * 255
        ).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

        top_limit = int(COLOR_HEIGHT * TOP_REGION_FRACTION)
        hough_debug_image = np.zeros((COLOR_HEIGHT, COLOR_WIDTH, 3), dtype=np.uint8)

        if DRAW_TOP_REGION_LINE:
            cv2.line(display_image, (0, top_limit), (COLOR_WIDTH, top_limit), TOP_REGION_COLOR, 2)
            cv2.line(depth_colormap, (0, top_limit), (COLOR_WIDTH, top_limit), (255, 255, 255), 2)

        # Main Hough-line detector in top half.
        candidates, raw_hough_lines, reject_reasons, canny_edges = detect_branch_candidates(
            color_image,
            depth_image,
            COLOR_FOCAL_LENGTH_PX,
            runtime_params,
        )

        active_rejects = {k: v for k, v in reject_reasons.items() if v > 0}
        if active_rejects:
            print(f"Rejects: {active_rejects}  |  Accepted: {len(candidates)}")

        best_candidate = max(candidates, key=lambda c: c["score"]) if candidates else None

        hough_debug_image = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

        for line in raw_hough_lines:
            x1, y1, x2, y2 = line
            cv2.line(hough_debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw final accepted branch candidates with distance color.
        for candidate in candidates:
            color = depth_to_branch_color(candidate["depth_m"], runtime_params)
            thickness = 4 if candidate is best_candidate else 2

            cv2.polylines(display_image, [candidate["box"]], True, color, thickness)

            cx, cy = candidate["center"]
            cv2.circle(display_image, (int(cx), int(cy)), 4, color, -1)

            x, y, w, h = cv2.boundingRect(candidate["box"])
            cv2.putText(
                display_image,
                f"{candidate['depth_m']:.2f}m",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        close_count = sum(
            1 for c in candidates if c["depth_m"] <= runtime_params["green_threshold_m"]
        )

        cv2.putText(
            display_image,
            f"Line-pair candidates: {len(candidates)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_image,
            f"Close enough (green): {close_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            hough_debug_image,
            f"Hough lines: {len(raw_hough_lines)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if best_candidate is not None:
            cv2.putText(
                display_image,
                f"Best depth: {best_candidate['depth_m']:.2f} m",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                depth_to_branch_color(best_candidate["depth_m"], runtime_params),
                2,
            )
            cv2.putText(
                display_image,
                f"Length: {best_candidate['estimated_length_m'] * 39.3701:.1f} in",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            depth_colormap,
            (
                f"Depth range: {runtime_params['min_depth_m']:.2f}m to "
                f"{runtime_params['max_depth_m']:.2f}m"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            depth_colormap,
            (
                f"Green {runtime_params['green_threshold_m']:.2f}m  "
                f"Bin {runtime_params['depth_bin_mm']}mm"
            ),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            depth_colormap,
            (
                f"Valid {runtime_params['min_valid_ratio']:.2f}  "
                f"Majority {runtime_params['min_majority_ratio']:.2f}"
            ),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            hough_debug_image,
            "Canny + Hough Overlay",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            hough_debug_image,
            (
                f"Canny {runtime_params['canny_low']}/{runtime_params['canny_high']}  "
                f"Hough {runtime_params['hough_threshold']}"
            ),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            hough_debug_image,
            (
                f"MinLen {runtime_params['hough_min_line_length']}  "
                f"GapRef {runtime_params['gap_ref_px']}  "
                f"GapRange {runtime_params['gap_min_px']}-{runtime_params['gap_max_px']}"
            ),
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )

        combined = np.hstack((display_image, depth_colormap, hough_debug_image))

        cv2.imshow(MAIN_WINDOW_NAME, combined)
        render_control_window(
            CANNY_CONTROL_WINDOW_NAME,
            "Canny Controls",
            [
                ("Canny Low", runtime_params["canny_low"]),
                ("Canny High", runtime_params["canny_high"]),
                ("CLAHE x10", int(round(runtime_params["clahe_clip_limit"] * 10.0))),
            ],
            width=520,
            row_height=42,
        )
        render_control_window(
            CONTROL_WINDOW_NAME,
            "Raw Hough Controls",
            [
                ("Hough Threshold", runtime_params["hough_threshold"]),
                ("Min Line Len", runtime_params["hough_min_line_length"]),
                ("Gap Ref Px", runtime_params["gap_ref_px"]),
                ("Gap Min Px", runtime_params["gap_min_px"]),
                ("Gap Max Px", runtime_params["gap_max_px"]),
            ],
            width=520,
            row_height=38,
        )
        render_control_window(
            DEPTH_CONTROL_WINDOW_NAME,
            "Depth Controls",
            [
                ("Min Depth cm", int(round(runtime_params["min_depth_m"] * 100.0))),
                ("Max Depth cm", int(round(runtime_params["max_depth_m"] * 100.0))),
                ("Green cm", int(round(runtime_params["green_threshold_m"] * 100.0))),
                ("Depth Bin mm", runtime_params["depth_bin_mm"]),
                ("Min Valid %", int(round(runtime_params["min_valid_ratio"] * 100.0))),
                ("Min Majority %", int(round(runtime_params["min_majority_ratio"] * 100.0))),
            ],
            width=520,
            row_height=36,
        )
        render_control_window(
            CAMERA_CONTROL_WINDOW_NAME,
            "Camera Controls",
            [
                ("Visual Preset", runtime_params["visual_preset"]),
                ("Emitter", runtime_params["emitter_enabled"]),
                ("Laser Power", runtime_params["laser_power"]),
                ("Auto Exposure", runtime_params["auto_exposure"]),
                ("Exposure", runtime_params["exposure"]),
                ("Gain", runtime_params["gain"]),
                ("Spatial Mag", runtime_params["spatial_magnitude"]),
                ("Spatial Alpha", runtime_params["spatial_alpha"]),
                ("Spatial Delta", runtime_params["spatial_delta"]),
                ("Temporal Alpha", runtime_params["temporal_alpha"]),
                ("Temporal Delta", runtime_params["temporal_delta"]),
            ],
            width=560,
            row_height=32,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()





