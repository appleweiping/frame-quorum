"""Small independent arithmetic oracle for the reused five-detector contract.

No production decision, rolling-window, fade, distance or quorum helper is used
to construct expectations. The production helpers are only the test subject.
This is bounded metric evidence, not real-media accuracy or general equivalence.
"""

import math
import random
from itertools import pairwise

from frame_quorum import FrameMetrics


def hand_content(left, right):
    bits = sum((left.perceptual_hash >> bit & 1) != (right.perceptual_hash >> bit & 1) for bit in range(64))
    squared = sum(
        (getattr(left, name) - getattr(right, name)) ** 2 for name in ("mean_red", "mean_green", "mean_blue")
    )
    return min(
        1.0, 0.65 * bits / 64 + 0.25 * math.sqrt(squared / 3) + 0.1 * abs(left.luminance - right.luminance)
    )


def hand_detector(metrics, config):
    count = len(metrics)
    content = [0.0] + [hand_content(metrics[i - 1], metrics[i]) for i in range(1, count)] if count else []
    candidates = {}
    if config.detector == "adaptive":
        scores = [None] * count
        radius = config.window_radius
        for position in range(radius + 1, count - radius):
            neighbors = [
                content[index]
                for index in range(position - radius, position + radius + 1)
                if index != position
            ]
            baseline = math.fsum(neighbors) / (2 * radius)
            scores[position] = min(1_000_000.0, content[position] / max(1e-12, baseline))
            if scores[position] >= config.adaptive_ratio and content[position] >= config.min_content:
                candidates[position] = "adaptive_peak"
    elif config.detector == "threshold":
        scores = [item.luminance for item in metrics]
        # Enumerate each interval between bright releases directly, not through
        # production's rolling armed/start/count state machine.
        bright = [i for i, value in enumerate(scores) if value > config.dark_threshold + config.hysteresis]
        for left, right in pairwise(bright):
            dark = [i for i in range(left + 1, right) if scores[i] <= config.dark_threshold]
            if len(dark) >= config.min_dark_frames:
                start = dark[0]
                index = start + math.floor((right - start) * (1 + config.fade_bias) / 2)
                candidates[index] = "completed_fade"
        if bright and config.include_final_fade:
            tail = [i for i in range(bright[-1] + 1, count) if scores[i] <= config.dark_threshold]
            if len(tail) >= config.min_dark_frames:
                candidates[tail[0]] = "final_fade"
    else:
        scores = content[:]
        for position in range(1, count):
            left, right = metrics[position - 1], metrics[position]
            if config.detector == "luminance":
                scores[position] = abs(left.luminance - right.luminance)
            elif config.detector == "color":
                scores[position] = math.sqrt(
                    sum(
                        (getattr(left, channel) - getattr(right, channel)) ** 2
                        for channel in ("mean_red", "mean_green", "mean_blue")
                    )
                    / 3
                )
            if scores[position] >= config.threshold:
                candidates[position] = "distance_threshold"
    kept = []
    decisions = []
    for position, score in enumerate(scores):
        if position in candidates:
            previous = kept[-1] if kept else 0
            if position - previous < config.min_scene_frames:
                outcome = False, "short_previous_scene"
            elif count - position < config.min_scene_frames:
                outcome = False, "short_final_scene"
            else:
                outcome = True, candidates[position]
                kept.append(position)
        elif position == 0:
            outcome = False, "sequence_start"
        elif config.detector == "threshold":
            outcome = False, "no_completed_fade"
        else:
            outcome = False, "incomplete_window" if score is None else "below_threshold"
        decisions.append(outcome)
    return content, scores, candidates, decisions


def records(pattern):
    generator = random.Random(3407 + pattern)
    sequences = [
        [],
        [1.0],
        [0.0] * 15,
        [1.0] * 15,
        [1.0, 1.0, 0.0, 0.0, 0.06, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.04, 0.06, 0.0, 1.0, 0.0, 0.06, 0.0, 0.08],
        [0.0, 1.0] * 9,
    ]
    values = (
        sequences[pattern]
        if pattern < len(sequences)
        else [generator.choice((0.0, 0.04, 0.06, 0.25, 0.5, 1.0)) for _ in range(21)]
    )
    if pattern < 12:
        return [FrameMetrics(0, value, 0.0, 0.0, 0.0, value, value, value) for value in values]
    return [
        FrameMetrics(
            generator.getrandbits(64),
            value,
            0.0,
            0.0,
            0.0,
            *[generator.choice((0.0, 0.25, 0.5, 0.75, 1.0)) for _ in range(3)],
        )
        for value in values
    ]
