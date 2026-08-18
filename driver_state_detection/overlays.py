import math
from dataclasses import dataclass

import cv2
import numpy as np

PANEL_COLOR = (24, 27, 32)
PANEL_BORDER = (75, 82, 90)
TEXT_COLOR = (242, 244, 246)
MUTED_COLOR = (155, 165, 175)
TRACK_COLOR = (62, 68, 76)
SAFE_COLOR = (92, 196, 126)
WARNING_COLOR = (0, 196, 255)
DANGER_COLOR = (68, 74, 235)
ZERO_COLOR = (235, 238, 240)


@dataclass(frozen=True)
class SignedBarGeometry:
    center_x: int
    value_x: int
    negative_threshold_x: int
    positive_threshold_x: int
    display_limit: float


def _finite(value):
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def format_ratio(value):
    """Format EAR and gaze ratios with useful, non-noisy precision."""
    value = _finite(value)
    return "--" if value is None else f"{value:.2f}"


def format_percent(value):
    """Format a zero-to-one ratio as a clipped whole percentage."""
    value = _finite(value)
    return "--" if value is None else f"{np.clip(value, 0.0, 1.0) * 100:.0f}%"


def format_angle(value):
    """Format a signed angle while avoiding a confusing negative zero."""
    value = _finite(value)
    if value is None:
        return "--"
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:+.1f} deg"


def _status_color(value, threshold, danger_below=False, danger_at_threshold=False):
    value = _finite(value)
    if value is None:
        return MUTED_COLOR
    magnitude = value if danger_below else abs(value)
    if danger_below:
        return DANGER_COLOR if magnitude <= threshold else SAFE_COLOR
    if magnitude > threshold or (danger_at_threshold and magnitude >= threshold):
        return DANGER_COLOR
    if magnitude >= threshold * 0.75:
        return WARNING_COLOR
    return SAFE_COLOR


def _signed_bar_geometry(value, threshold, x, width, display_limit):
    """Map a signed value and symmetric thresholds onto a horizontal track."""
    if width <= 0:
        raise ValueError("bar width must be positive")
    threshold = abs(float(threshold))
    display_limit = float(display_limit)
    if display_limit <= 0 or threshold > display_limit:
        raise ValueError("bar limit must be positive and include its threshold")
    center_x = x + width // 2
    half_width = width / 2
    value = _finite(value) or 0.0
    clipped_value = float(np.clip(value, -display_limit, display_limit))
    value_x = round(center_x + clipped_value / display_limit * half_width)
    threshold_offset = round(min(threshold, display_limit) / display_limit * half_width)
    return SignedBarGeometry(
        center_x=center_x,
        value_x=int(np.clip(value_x, x, x + width)),
        negative_threshold_x=center_x - threshold_offset,
        positive_threshold_x=center_x + threshold_offset,
        display_limit=display_limit,
    )


def _blend_panel(frame, x, y, width, height, alpha=0.72):
    """Blend a dark panel into a clipped ROI instead of copying the full frame."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_width, x + width), min(frame_height, y + height)
    if x1 >= x2 or y1 >= y2:
        return
    roi = frame[y1:y2, x1:x2]
    panel = np.full_like(roi, PANEL_COLOR)
    cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0, dst=roi)
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), PANEL_BORDER, 1)


def _put_text(frame, text, origin, scale, color=TEXT_COLOR, thickness=1):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _put_right_aligned(frame, text, right_x, baseline_y, scale, color):
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    _put_text(frame, text, (right_x - text_width, baseline_y), scale, color)


def _draw_attention_panel(
    frame,
    rect,
    scale,
    ear,
    gaze,
    perclos,
    perclos_ready,
    ear_threshold,
    gaze_threshold,
    perclos_threshold,
):
    x, y, width, height = rect
    _blend_panel(frame, *rect)
    padding = max(6, round(10 * scale))
    text_scale = max(0.35, 0.48 * scale)
    _put_text(frame, "ATTENTION", (x + padding, y + round(22 * scale)), text_scale)

    rows = (
        (
            "EAR",
            format_ratio(ear),
            _status_color(ear, ear_threshold, danger_below=True),
        ),
        ("GAZE", format_ratio(gaze), _status_color(gaze, gaze_threshold)),
        (
            "PERCLOS",
            format_percent(perclos),
            (
                WARNING_COLOR
                if not perclos_ready
                else _status_color(perclos, perclos_threshold, danger_at_threshold=True)
            ),
        ),
    )
    first_row_y = y + round(48 * scale)
    row_gap = max(16, round(27 * scale))
    for index, (label, value_text, color) in enumerate(rows):
        baseline = first_row_y + index * row_gap
        _put_text(frame, label, (x + padding, baseline), text_scale, MUTED_COLOR)
        if label == "PERCLOS" and not perclos_ready and width >= 180:
            value_text = f"{value_text}  WARMING"
        _put_right_aligned(
            frame, value_text, x + width - padding, baseline, text_scale, color
        )


def _draw_signed_bar(frame, x, y, width, label, value, threshold, display_limit, scale):
    text_scale = max(0.32, 0.43 * scale)
    value_text = format_angle(value)
    color = _status_color(value, threshold)
    _put_text(frame, label.upper(), (x, y), text_scale, MUTED_COLOR)
    _put_right_aligned(frame, value_text, x + width, y, text_scale, color)

    track_y = y + max(5, round(8 * scale))
    track_height = max(5, round(8 * scale))
    geometry = _signed_bar_geometry(value, threshold, x, width, display_limit)
    cv2.rectangle(
        frame,
        (x, track_y),
        (x + width, track_y + track_height),
        TRACK_COLOR,
        cv2.FILLED,
    )
    if _finite(value) is not None:
        cv2.rectangle(
            frame,
            (min(geometry.center_x, geometry.value_x), track_y),
            (max(geometry.center_x, geometry.value_x), track_y + track_height),
            color,
            cv2.FILLED,
        )

    marker_top = track_y - 2
    marker_bottom = track_y + track_height + 2
    for marker_x in (
        geometry.negative_threshold_x,
        geometry.positive_threshold_x,
    ):
        cv2.line(
            frame,
            (marker_x, marker_top),
            (marker_x, marker_bottom),
            WARNING_COLOR,
            1,
        )
    cv2.line(
        frame,
        (geometry.center_x, marker_top),
        (geometry.center_x, marker_bottom),
        ZERO_COLOR,
        1,
    )
    cv2.rectangle(
        frame,
        (x, track_y),
        (x + width, track_y + track_height),
        PANEL_BORDER,
        1,
    )


def _draw_pose_panel(
    frame,
    rect,
    scale,
    roll,
    pitch,
    yaw,
    roll_threshold,
    pitch_threshold,
    yaw_threshold,
    roll_limit,
    pitch_limit,
    yaw_limit,
):
    x, y, width, height = rect
    _blend_panel(frame, *rect)
    padding = max(6, round(10 * scale))
    text_scale = max(0.35, 0.48 * scale)
    _put_text(frame, "HEAD POSE", (x + padding, y + round(22 * scale)), text_scale)
    bar_x = x + padding
    bar_width = width - 2 * padding
    first_row_y = y + round(47 * scale)
    row_gap = max(25, round(46 * scale))
    rows = (
        ("Roll", roll, roll_threshold, roll_limit),
        ("Pitch", pitch, pitch_threshold, pitch_limit),
        ("Yaw", yaw, yaw_threshold, yaw_limit),
    )
    for index, (label, value, threshold, limit) in enumerate(rows):
        _draw_signed_bar(
            frame,
            bar_x,
            first_row_y + index * row_gap,
            bar_width,
            label,
            value,
            threshold,
            limit,
            scale,
        )


def _draw_badge(frame, text, center_x, y, scale, color):
    text_scale = max(0.38, 0.48 * scale)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1
    )
    padding_x, padding_y = round(8 * scale), round(5 * scale)
    x1 = max(0, center_x - text_width // 2 - padding_x)
    y1 = max(0, y - text_height - padding_y)
    x2 = min(frame.shape[1] - 1, center_x + text_width // 2 + padding_x)
    y2 = min(frame.shape[0] - 1, y + baseline + padding_y)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PANEL_COLOR, cv2.FILLED)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    _put_text(frame, text, (center_x - text_width // 2, y), text_scale, color)


def _draw_compact_dashboard(
    frame, ear, gaze, perclos, roll, pitch, yaw, perclos_ready, margin
):
    """Keep essential values legible when the full signed bars cannot fit."""
    width = frame.shape[1] - 2 * margin
    height = min(frame.shape[0] - 2 * margin, 52)
    _blend_panel(frame, margin, margin, width, height)
    attention = (
        f"EAR {format_ratio(ear)}  G {format_ratio(gaze)}  "
        f"P {format_percent(perclos)}"
    )
    if not perclos_ready:
        attention += " WARM"
    pose = (
        f"R {format_angle(roll).removesuffix(' deg')}  "
        f"P {format_angle(pitch).removesuffix(' deg')}  "
        f"Y {format_angle(yaw).removesuffix(' deg')}"
    )
    _put_text(frame, attention, (margin + 5, margin + 19), 0.34)
    _put_text(frame, pose, (margin + 5, margin + 41), 0.34, MUTED_COLOR)
    return margin + height


def draw_dashboard(
    frame,
    *,
    ear,
    gaze,
    perclos,
    perclos_ready,
    roll,
    pitch,
    yaw,
    ear_threshold,
    gaze_threshold,
    perclos_threshold,
    roll_threshold,
    pitch_threshold,
    yaw_threshold,
    roll_limit,
    pitch_limit,
    yaw_limit,
    alerts=(),
    face_detected=True,
    fps=None,
    processing_ms=None,
    show_panels=True,
):
    """Draw a responsive two-panel dashboard directly onto ``frame``."""
    frame_height, frame_width = frame.shape[:2]
    scale = float(np.clip(min(frame_width / 640, frame_height / 480), 0.55, 1.25))
    margin = max(4, round(min(frame_width, frame_height) * 0.015))
    gap = max(5, round(9 * scale))
    compact = show_panels and (frame_width < 280 or frame_height < 220)
    if compact:
        panels_bottom = _draw_compact_dashboard(
            frame,
            ear,
            gaze,
            perclos,
            roll,
            pitch,
            yaw,
            perclos_ready,
            margin,
        )
        show_panels = False

    attention_width = round(205 * scale)
    attention_height = round(118 * scale)
    pose_width = round(345 * scale)
    pose_height = round(184 * scale)

    if not show_panels:
        attention_rect = pose_rect = (0, 0, 0, 0)
    elif margin * 2 + attention_width + gap + pose_width <= frame_width:
        attention_rect = (margin, margin, attention_width, attention_height)
        pose_rect = (
            frame_width - margin - pose_width,
            margin,
            pose_width,
            pose_height,
        )
        panels_bottom = margin + max(attention_height, pose_height)
    else:
        panel_width = max(1, frame_width - 2 * margin)
        attention_rect = (margin, margin, panel_width, attention_height)
        pose_rect = (
            margin,
            margin + attention_height + gap,
            panel_width,
            pose_height,
        )
        panels_bottom = pose_rect[1] + pose_height

    if show_panels:
        _draw_attention_panel(
            frame,
            attention_rect,
            scale,
            ear,
            gaze,
            perclos,
            perclos_ready,
            ear_threshold,
            gaze_threshold,
            perclos_threshold,
        )
        _draw_pose_panel(
            frame,
            pose_rect,
            scale,
            roll,
            pitch,
            yaw,
            roll_threshold,
            pitch_threshold,
            yaw_threshold,
            roll_limit,
            pitch_limit,
            yaw_limit,
        )
    elif not compact:
        panels_bottom = margin

    badges = list(alerts)
    if not face_detected:
        badges.insert(0, "FACE NOT DETECTED")
    badge_y = min(frame_height - round(45 * scale), panels_bottom + round(24 * scale))
    for index, badge in enumerate(badges):
        _draw_badge(
            frame,
            badge,
            frame_width // 2,
            badge_y + index * max(18, round(25 * scale)),
            scale,
            WARNING_COLOR if badge == "FACE NOT DETECTED" else DANGER_COLOR,
        )

    performance = []
    if _finite(fps) is not None:
        performance.append(f"FPS {fps:.1f}")
    if _finite(processing_ms) is not None:
        performance.append(f"Processing {processing_ms:.1f} ms")
    if performance:
        _put_text(
            frame,
            "  |  ".join(performance),
            (margin, frame_height - margin),
            max(0.32, 0.42 * scale),
            TEXT_COLOR,
        )

    return frame
