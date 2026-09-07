"""Run independent synthetic cut/fade examples; no files or downloads required."""

from pathlib import Path

from frame_quorum import DetectionConfig, Frame, FrameMetrics, detect_scenes


def sequence(values: list[float]) -> tuple[Frame, ...]:
    return tuple(
        Frame(
            index,
            Path(f"synthetic-{index}.png"),
            f"synthetic-{index}.png",
            index / 25,
            16,
            16,
            0,
            FrameMetrics(0, level, 0, 0, 0, level, level, level),
        )
        for index, level in enumerate(values)
    )


adaptive = detect_scenes(sequence([0, 0.05, 0.1, 0.15, 0.8, 0.85, 0.9, 0.95, 1]))
assert adaptive.cut_indices == (4,)
print("Adaptive cut:", adaptive.cut_indices, "contrast ratio:", adaptive.statistics[4].detector_score)

fade = detect_scenes(sequence([1, 1, 0, 0, 0, 0, 1, 1]), DetectionConfig(detector="threshold"))
assert fade.cut_indices == (4,)
print("Fade cut:", fade.cut_indices, "scene lengths:", [scene.frame_count for scene in fade.scenes])
