import cv2
import numpy as np
import pyrealsense2 as rs


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
# Main contour detector region
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

# Used to shrink contour mask before depth voting.
MASK_ERODE_KERNEL = (5, 3)

# -----------------------------
# Contour / rotated-rect filters
# -----------------------------
MIN_CONTOUR_AREA = 400
MIN_MAJOR_AXIS_PX = 60
MIN_ASPECT_RATIO = 2.8
MAX_HORIZONTAL_DEVIATION_DEG = 15.0
MIN_SOLIDITY = 0.35
MIN_FILL_RATIO = 0.12
MAX_FILL_RATIO = 0.95

# -----------------------------
# Depth voting
# -----------------------------
DEPTH_BIN_MM = 20
MIN_VALID_DEPTH_RATIO = 0.05
MIN_MAJORITY_RATIO = 0.18

# -----------------------------
# Debug Hough / parallel-line overlay
# This is only for visualization.
# -----------------------------
DEBUG_SHOW_PARALLEL_LINES = True
DEBUG_CANNY_LOW = 50
DEBUG_CANNY_HIGH = 150
DEBUG_HOUGH_THRESHOLD = 45
DEBUG_MIN_LINE_LENGTH = 100
DEBUG_MAX_LINE_GAP = 20
DEBUG_HORIZONTAL_TOLERANCE_DEG = 10.0
DEBUG_MAX_PAIR_ANGLE_DIFF_DEG = 20.0
DEBUG_MAX_LINE_DISTANCE_PX = 150.0
DEBUG_MIN_HORIZONTAL_OVERLAP_PX = 40.0
DEBUG_PAIR_PAD_X = 10
DEBUG_PAIR_PAD_Y = 6

# -----------------------------
# Display / debugging
# -----------------------------
DRAW_TOP_THIRD_LINE = True
DRAW_BINARY_VIEW = True
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


def contour_score(candidate, image_width):
    center_x = candidate["center"][0]
    center_bonus = 1.0 - abs(center_x - image_width / 2.0) / (image_width / 2.0)
    center_bonus = np.clip(center_bonus, 0.0, 1.0)

    horizontal_bonus = 1.0 - candidate["horizontal_dev_deg"] / MAX_HORIZONTAL_DEVIATION_DEG
    horizontal_bonus = np.clip(horizontal_bonus, 0.0, 1.0)

    aspect_bonus = np.clip(candidate["aspect_ratio"] / 10.0, 0.0, 1.0)
    area_bonus = np.clip(candidate["area"] / 5000.0, 0.0, 1.0)

    return (
        0.30 * center_bonus
        + 0.25 * horizontal_bonus
        + 0.20 * aspect_bonus
        + 0.15 * candidate["majority_ratio"]
        + 0.10 * area_bonus
    )


def detect_debug_parallel_pairs(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, DEBUG_CANNY_LOW, DEBUG_CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=DEBUG_HOUGH_THRESHOLD,
        minLineLength=DEBUG_MIN_LINE_LENGTH,
        maxLineGap=DEBUG_MAX_LINE_GAP,
    )

    if lines is None:
        return [], [], edges

    line_segments = [tuple(line[0]) for line in lines]
    line_angles = [
        line_angle_degrees(x1, y1, x2, y2)
        for x1, y1, x2, y2 in line_segments
    ]
    line_midpoints = [
        line_midpoint(x1, y1, x2, y2)
        for x1, y1, x2, y2 in line_segments
    ]

    qualifying_line_indexes = set()
    pair_regions = []

    for i in range(len(line_segments)):
        if horizontal_deviation_degrees(line_angles[i]) > DEBUG_HORIZONTAL_TOLERANCE_DEG:
            continue

        for j in range(i + 1, len(line_segments)):
            if horizontal_deviation_degrees(line_angles[j]) > DEBUG_HORIZONTAL_TOLERANCE_DEG:
                continue

            angle_diff = angle_difference_degrees(line_angles[i], line_angles[j])
            if angle_diff > DEBUG_MAX_PAIR_ANGLE_DIFF_DEG:
                continue

            midpoint_distance = np.hypot(
                line_midpoints[i][0] - line_midpoints[j][0],
                line_midpoints[i][1] - line_midpoints[j][1],
            )
            if midpoint_distance > DEBUG_MAX_LINE_DISTANCE_PX:
                continue

            overlap_px = horizontal_overlap(line_segments[i], line_segments[j])
            if overlap_px < DEBUG_MIN_HORIZONTAL_OVERLAP_PX:
                continue

            qualifying_line_indexes.add(i)
            qualifying_line_indexes.add(j)

            ax1, ay1, ax2, ay2 = line_segments[i]
            bx1, by1, bx2, by2 = line_segments[j]

            region = clamp_bbox(
                (
                    min(ax1, ax2, bx1, bx2) - DEBUG_PAIR_PAD_X,
                    min(ay1, ay2, by1, by2) - DEBUG_PAIR_PAD_Y,
                    max(ax1, ax2, bx1, bx2) + DEBUG_PAIR_PAD_X,
                    max(ay1, ay2, by1, by2) + DEBUG_PAIR_PAD_Y,
                ),
                color_image.shape[1],
                color_image.shape[0],
            )
            pair_regions.append(region)

    debug_lines = [line_segments[i] for i in qualifying_line_indexes]
    return debug_lines, pair_regions, edges


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
        "area": 0, "hull": 0, "solidity": 0, "minor_axis": 0,
        "major_axis": 0, "aspect": 0, "fill": 0, "angle": 0,
        "depth_none": 0, "valid_ratio": 0, "majority_ratio": 0,
        "depth_range": 0,
    }

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            reject_reasons["area"] += 1
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            reject_reasons["hull"] += 1
            continue

        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            reject_reasons["solidity"] += 1
            continue

        rect = cv2.minAreaRect(contour)
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

        rect_area = max(major_axis * minor_axis, 1.0)
        fill_ratio = area / rect_area
        if fill_ratio < MIN_FILL_RATIO or fill_ratio > MAX_FILL_RATIO:
            reject_reasons["fill"] += 1
            continue

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        horizontal_angle = longest_edge_angle_deg(box)
        horizontal_dev = horizontal_deviation_degrees(horizontal_angle)
        if horizontal_dev > MAX_HORIZONTAL_DEVIATION_DEG:
            reject_reasons["angle"] += 1
            continue

        # Depth vote on a shrunken contour mask so nearby clutter contributes less.
        mask = np.zeros((top_limit, image_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
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

        candidate = {
            "contour": contour,
            "box": box,
            "center": (float(cx), float(cy)),
            "area": float(area),
            "major_axis": float(major_axis),
            "minor_axis": float(minor_axis),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
            "fill_ratio": float(fill_ratio),
            "horizontal_dev_deg": float(horizontal_dev),
            "depth_m": depth_m,
            "valid_ratio": depth_vote["valid_ratio"],
            "majority_ratio": depth_vote["majority_ratio"],
            "mask": full_mask,
        }
        candidate["score"] = contour_score(candidate, image_w)
        candidates.append(candidate)

    return candidates, binary, reject_reasons


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

        # Debug parallel-line overlay across the full frame.
        if DEBUG_SHOW_PARALLEL_LINES:
            debug_lines, debug_regions, debug_edges = detect_debug_parallel_pairs(color_image)

            for line in debug_lines:
                x1, y1, x2, y2 = line
                cv2.line(display_image, (x1, y1), (x2, y2), YELLOW, 2)

            for region in debug_regions:
                x1, y1, x2, y2 = region
                cv2.rectangle(display_image, (x1, y1), (x2, y2), YELLOW, 1)
        else:
            debug_edges = np.zeros((COLOR_HEIGHT, COLOR_WIDTH), dtype=np.uint8)

        # Main contour-based detector in top third.
        candidates, binary, reject_reasons = detect_branch_candidates(color_image, depth_image)

        active_rejects = {k: v for k, v in reject_reasons.items() if v > 0}
        if active_rejects:
            print(f"Rejects: {active_rejects}  |  Accepted: {len(candidates)}")

        best_candidate = max(candidates, key=lambda c: c["score"]) if candidates else None

        # Draw all accepted contour regions in yellow first.
        for candidate in candidates:
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
            f"Contour candidates: {len(candidates)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_image,
            f"Debug parallel lines: {len(debug_lines) if DEBUG_SHOW_PARALLEL_LINES else 0}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            YELLOW,
            2,
        )
        cv2.putText(
            display_image,
            f"Close enough (green): {close_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        if best_candidate is not None:
            cv2.putText(
                display_image,
                f"Best depth: {best_candidate['depth_m']:.2f} m",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                depth_to_branch_color(best_candidate["depth_m"]),
                2,
            )
            cv2.putText(
                display_image,
                f"Aspect: {best_candidate['aspect_ratio']:.1f}  Solidity: {best_candidate['solidity']:.2f}",
                (10, 150),
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

        if DRAW_BINARY_VIEW:
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            debug_edges_bgr = cv2.cvtColor(debug_edges, cv2.COLOR_GRAY2BGR)

            binary_bgr = cv2.resize(binary_bgr, (COLOR_WIDTH, COLOR_HEIGHT))
            debug_edges_bgr = cv2.resize(debug_edges_bgr, (COLOR_WIDTH, COLOR_HEIGHT))

            cv2.putText(binary_bgr, "Contour Binary", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
            cv2.putText(debug_edges_bgr, "Debug Hough Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            combined = np.vstack(
                (
                    np.hstack((display_image, depth_colormap)),
                    np.hstack((binary_bgr, debug_edges_bgr)),
                )
            )
        else:
            combined = np.hstack((display_image, depth_colormap))

        cv2.imshow("D435 Contour Branch Detection + Debug", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()





