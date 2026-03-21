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
GREEN_THRESHOLD_M = 0.1524  # 6 inches

MIN_DEPTH_MM = int(MIN_DEPTH_M * 1000)
MAX_DEPTH_MM = int(MAX_DEPTH_M * 1000)

# -----------------------------
# Main Hough-line detector region
# -----------------------------
TOP_REGION_FRACTION = 1.0 / 3.0

# -----------------------------
# RGB preprocessing
# -----------------------------
CANNY_LOW = 70
CANNY_HIGH = 170
GAUSSIAN_KERNEL = (5, 5)

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)

# Tighter morphology than before to avoid grabbing too much nearby clutter.
MORPH_CLOSE_KERNEL = (13, 5)
MORPH_OPEN_KERNEL = (5, 3)

# Used to tighten the filled line-pair mask before depth voting.
MASK_ERODE_KERNEL = (5, 3)

# -----------------------------
# Hough-line / line-pair filters
# -----------------------------
HOUGH_THRESHOLD = 35
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 20

MIN_PAIR_AREA = 250
MIN_MAJOR_AXIS_PX = 60
MIN_ASPECT_RATIO = 2.0
MAX_HORIZONTAL_DEVIATION_DEG = 15.0
MAX_PAIR_ANGLE_DIFF_DEG = 12.0
MAX_PAIR_DISTANCE_PX = 120.0
MIN_PAIR_VERTICAL_SEPARATION_PX = 8.0
MIN_HORIZONTAL_OVERLAP_PX = 40.0
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
DRAW_TOP_THIRD_LINE = True
YELLOW = (0, 255, 255)


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


def line_midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def x_range_of_line(line):
    x1, _, x2, _ = line
    return min(x1, x2), max(x1, x2)


def horizontal_overlap(line_a, line_b):
    a_min, a_max = x_range_of_line(line_a)
    b_min, b_max = x_range_of_line(line_b)
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


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


def dominant_depth_from_mask(depth_image, mask):
    masked_pixels = depth_image[mask > 0]
    if masked_pixels.size == 0:
        return None

    valid_depth = masked_pixels[
        (masked_pixels >= MIN_DEPTH_MM) & (masked_pixels <= MAX_DEPTH_MM)
    ]
    if valid_depth.size == 0:
        return None

    valid_ratio = valid_depth.size / masked_pixels.size

    bins = np.arange(MIN_DEPTH_MM, MAX_DEPTH_MM + DEPTH_BIN_MM, DEPTH_BIN_MM)
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


def depth_to_branch_color(depth_m):
    if depth_m <= GREEN_THRESHOLD_M:
        return (0, 255, 0)

    normalized = (MAX_DEPTH_M - depth_m) / (MAX_DEPTH_M - GREEN_THRESHOLD_M)
    normalized = np.clip(normalized, 0.0, 1.0)
    blue_value = int(80 + normalized * (255 - 80))
    return (blue_value, 0, 0)


def candidate_score(candidate, image_width):
    center_x = candidate["center"][0]
    center_bonus = 1.0 - abs(center_x - image_width / 2.0) / (image_width / 2.0)
    center_bonus = np.clip(center_bonus, 0.0, 1.0)

    horizontal_bonus = 1.0 - candidate["horizontal_dev_deg"] / MAX_HORIZONTAL_DEVIATION_DEG
    horizontal_bonus = np.clip(horizontal_bonus, 0.0, 1.0)

    overlap_bonus = np.clip(candidate["overlap_px"] / 200.0, 0.0, 1.0)
    span_bonus = np.clip(candidate["major_axis"] / 250.0, 0.0, 1.0)

    return (
        0.30 * center_bonus
        + 0.25 * horizontal_bonus
        + 0.20 * overlap_bonus
        + 0.15 * candidate["majority_ratio"]
        + 0.10 * span_bonus
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


def detect_branch_candidates(color_image, depth_image):
    image_h, image_w = color_image.shape[:2]
    top_limit = int(image_h * TOP_REGION_FRACTION)

    search_region = color_image[:top_limit, :]
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MASK_ERODE_KERNEL)

    binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    reject_reasons = {
        "no_lines": 0, "short_line": 0, "angle": 0, "pair_angle": 0,
        "pair_distance": 0, "pair_separation": 0, "pair_overlap": 0,
        "pair_area": 0, "minor_axis": 0, "major_axis": 0, "aspect": 0,
        "depth_none": 0, "valid_ratio": 0, "majority_ratio": 0,
        "depth_range": 0,
    }

    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None:
        reject_reasons["no_lines"] += 1
        return candidates, contours, reject_reasons

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        length_px = line_length(x1, y1, x2, y2)
        if length_px < MIN_MAJOR_AXIS_PX:
            reject_reasons["short_line"] += 1
            continue

        angle = line_angle_degrees(x1, y1, x2, y2)
        horizontal_dev = horizontal_deviation_degrees(angle)
        if horizontal_dev > MAX_HORIZONTAL_DEVIATION_DEG:
            reject_reasons["angle"] += 1
            continue

        filtered_lines.append(
            {
                "segment": (x1, y1, x2, y2),
                "angle_deg": angle,
                "horizontal_dev_deg": horizontal_dev,
                "midpoint": line_midpoint(x1, y1, x2, y2),
                "length_px": length_px,
            }
        )

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

            midpoint_distance = np.hypot(
                line_a["midpoint"][0] - line_b["midpoint"][0],
                line_a["midpoint"][1] - line_b["midpoint"][1],
            )
            if midpoint_distance > MAX_PAIR_DISTANCE_PX:
                reject_reasons["pair_distance"] += 1
                continue

            vertical_separation = abs(line_a["midpoint"][1] - line_b["midpoint"][1])
            if vertical_separation < MIN_PAIR_VERTICAL_SEPARATION_PX:
                reject_reasons["pair_separation"] += 1
                continue

            overlap_px = horizontal_overlap(line_a["segment"], line_b["segment"])
            if overlap_px < MIN_HORIZONTAL_OVERLAP_PX:
                reject_reasons["pair_overlap"] += 1
                continue

            pair_points = np.array(
                [(ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2)],
                dtype=np.int32,
            )
            pair_hull = cv2.convexHull(pair_points)
            area = cv2.contourArea(pair_hull)
            if area < MIN_PAIR_AREA:
                reject_reasons["pair_area"] += 1
                continue

            rect = cv2.minAreaRect(pair_hull)
            (cx, cy), (w, h), _ = rect
            major_axis = max(w, h)
            minor_axis = min(w, h)

            if minor_axis <= 0:
                reject_reasons["minor_axis"] += 1
                continue
            if major_axis < MIN_MAJOR_AXIS_PX:
                reject_reasons["major_axis"] += 1
                continue

            aspect_ratio = major_axis / minor_axis
            if aspect_ratio < MIN_ASPECT_RATIO:
                reject_reasons["aspect"] += 1
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

            depth_vote = dominant_depth_from_mask(depth_image, full_mask)
            if depth_vote is None:
                reject_reasons["depth_none"] += 1
                continue

            if depth_vote["valid_ratio"] < MIN_VALID_DEPTH_RATIO:
                reject_reasons["valid_ratio"] += 1
                continue
            if depth_vote["majority_ratio"] < MIN_MAJORITY_RATIO:
                reject_reasons["majority_ratio"] += 1
                continue

            depth_m = depth_vote["depth_m"]
            if not (MIN_DEPTH_M <= depth_m <= MAX_DEPTH_M):
                reject_reasons["depth_range"] += 1
                continue

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
                "area": float(area),
                "major_axis": float(major_axis),
                "minor_axis": float(minor_axis),
                "aspect_ratio": float(aspect_ratio),
                "horizontal_dev_deg": float(horizontal_dev),
                "overlap_px": float(overlap_px),
                "midpoint_distance_px": float(midpoint_distance),
                "vertical_separation_px": float(vertical_separation),
                "depth_m": depth_m,
                "valid_ratio": depth_vote["valid_ratio"],
                "majority_ratio": depth_vote["majority_ratio"],
                "mask": full_mask,
            }
            candidate["score"] = candidate_score(candidate, image_w)
            candidates.append(candidate)

    candidates = suppress_overlapping_candidates(candidates)
    return candidates, contours, reject_reasons


pipeline, profile, active_profile = start_pipeline_with_fallback()
align = rs.align(rs.stream.color)

COLOR_WIDTH = active_profile["color_width"]
COLOR_HEIGHT = active_profile["color_height"]

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

        clipped = np.clip(depth_image, MIN_DEPTH_MM, MAX_DEPTH_MM)
        normalized = (
            (clipped - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM) * 255
        ).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

        top_limit = int(COLOR_HEIGHT * TOP_REGION_FRACTION)
        if DRAW_TOP_THIRD_LINE:
            cv2.line(display_image, (0, top_limit), (COLOR_WIDTH, top_limit), (255, 255, 0), 2)
            cv2.line(depth_colormap, (0, top_limit), (COLOR_WIDTH, top_limit), (255, 255, 255), 2)

        # Main Hough-line detector in top third.
        candidates, contours, reject_reasons = detect_branch_candidates(color_image, depth_image)

        active_rejects = {k: v for k, v in reject_reasons.items() if v > 0}
        if active_rejects:
            print(f"Rejects: {active_rejects}  |  Accepted: {len(candidates)}")

        best_candidate = max(candidates, key=lambda c: c["score"]) if candidates else None

        # Draw contour outlines in the search region for context.
        if contours:
            cv2.drawContours(display_image[:top_limit, :], contours, -1, YELLOW, 1)

        # Draw all accepted line pairs in yellow first.
        for candidate in candidates:
            for line in candidate["lines"]:
                x1, y1, x2, y2 = line
                cv2.line(display_image, (x1, y1), (x2, y2), YELLOW, 1)
            cv2.polylines(display_image, [candidate["box"]], True, YELLOW, 2)

        # Draw final accepted branch candidates with distance color.
        for candidate in candidates:
            color = depth_to_branch_color(candidate["depth_m"])
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

        close_count = sum(1 for c in candidates if c["depth_m"] <= GREEN_THRESHOLD_M)

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

        if best_candidate is not None:
            cv2.putText(
                display_image,
                f"Best depth: {best_candidate['depth_m']:.2f} m",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                depth_to_branch_color(best_candidate["depth_m"]),
                2,
            )
            cv2.putText(
                display_image,
                f"Aspect: {best_candidate['aspect_ratio']:.1f}  Overlap: {best_candidate['overlap_px']:.0f}px",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            depth_colormap,
            f"Depth range: {MIN_DEPTH_M:.2f}m to {MAX_DEPTH_M:.2f}m",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        combined = np.hstack((display_image, depth_colormap))

        cv2.imshow("D435 Hough Line Branch Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()





