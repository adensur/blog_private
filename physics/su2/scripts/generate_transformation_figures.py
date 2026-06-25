from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "linear-algebra"

Point = tuple[float, float]


def sample_segment(start: Point, end: Point, n: int = 80) -> list[Point]:
    return [
        (
            start[0] + (end[0] - start[0]) * i / (n - 1),
            start[1] + (end[1] - start[1]) * i / (n - 1),
        )
        for i in range(n)
    ]


def sample_polyline(
    points: list[Point], n_per_segment: int = 80, closed: bool = False
) -> list[Point]:
    pts = points + ([points[0]] if closed else [])
    sampled: list[Point] = []
    for i in range(len(pts) - 1):
        segment = sample_segment(pts[i], pts[i + 1], n_per_segment)
        sampled.extend(segment[:-1] if i < len(pts) - 2 else segment)
    return sampled


def unit_square_grid() -> list[list[Point]]:
    lines: list[list[Point]] = []
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for value in values:
        lines.append(sample_segment((-1.0, value), (1.0, value)))
        lines.append(sample_segment((value, -1.0), (value, 1.0)))
    return lines


def shape() -> tuple[list[Point], list[Point], list[Point]]:
    boundary = sample_polyline(
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        closed=True,
    )
    arrow = sample_polyline(
        [
            (-0.65, -0.35),
            (0.2, -0.35),
            (0.2, -0.68),
            (0.75, 0.0),
            (0.2, 0.68),
            (0.2, 0.35),
            (-0.65, 0.35),
            (-0.65, -0.35),
        ]
    )
    landmarks = [(-0.72, 0.72), (0.72, 0.72), (0.72, -0.72)]
    return boundary, arrow, landmarks


def identity(point: Point) -> Point:
    return point


def linear_map(point: Point) -> Point:
    x, y = point
    return 1.15 * x + 0.55 * y, -0.25 * x + 0.9 * y


def nonlinear_map(point: Point) -> Point:
    x, y = point
    radius2 = x * x + y * y
    angle = 0.75 * radius2 + 0.25 * x
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    warped_x = cos_angle * x - sin_angle * y
    warped_y = sin_angle * x + cos_angle * y
    warped_y += 0.18 * math.sin(math.pi * x)
    return warped_x, warped_y


def points_attr(points: list[Point], x0: float, y0: float, size: float) -> str:
    min_world = -1.85
    max_world = 1.85
    scale = size / (max_world - min_world)

    def project(point: Point) -> Point:
        x, y = point
        return x0 + (x - min_world) * scale, y0 + (max_world - y) * scale

    return " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, points))


def circle_attrs(point: Point, x0: float, y0: float, size: float) -> tuple[float, float]:
    min_world = -1.85
    max_world = 1.85
    scale = size / (max_world - min_world)
    x, y = point
    return x0 + (x - min_world) * scale, y0 + (max_world - y) * scale


def transformed(points: list[Point], transform) -> list[Point]:
    return [transform(point) for point in points]


def panel(x0: float, y0: float, title: str, transform) -> str:
    size = 330.0
    boundary, arrow, landmarks = shape()
    colors = ["#d64045", "#f4a261", "#2a9d8f"]
    parts = [
        f'<text x="{x0 + size / 2:.1f}" y="{y0 - 25:.1f}" '
        'class="panel-title" text-anchor="middle">'
        f"{title}</text>",
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{size:.1f}" height="{size:.1f}" '
        'rx="4" class="frame" />',
    ]

    for line in unit_square_grid():
        parts.append(
            f'<polyline points="{points_attr(transformed(line, transform), x0, y0, size)}" '
            'class="grid-line" />'
        )

    parts.extend(
        [
            f'<polyline points="{points_attr([transform((-1.85, 0)), transform((1.85, 0))], x0, y0, size)}" '
            'class="axis-line" />',
            f'<polyline points="{points_attr([transform((0, -1.85)), transform((0, 1.85))], x0, y0, size)}" '
            'class="axis-line" />',
            f'<polyline points="{points_attr(transformed(boundary, transform), x0, y0, size)}" '
            'class="boundary" />',
            f'<polyline points="{points_attr(transformed(arrow, transform), x0, y0, size)}" '
            'class="arrow" />',
        ]
    )

    for landmark, color in zip(landmarks, colors):
        cx, cy = circle_attrs(transform(landmark), x0, y0, size)
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="6.5" '
            f'fill="{color}" stroke="#ffffff" stroke-width="1.5" />'
        )

    return "\n".join(parts)


def svg_document() -> str:
    panels = [
        panel(45, 90, "Original set", identity),
        panel(435, 90, "Linear map", linear_map),
        panel(825, 90, "Non-linear map", nonlinear_map),
    ]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 460" role="img" aria-labelledby="title desc">
  <title id="title">A shape transformed by linear and non-linear maps</title>
  <desc id="desc">Three panels show the same square, grid, arrow, and landmark points before transformation, after a linear map, and after a non-linear map.</desc>
  <style>
    .figure-title {{ font: 700 22px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #16181d; }}
    .panel-title {{ font: 650 17px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #222630; }}
    .frame {{ fill: #ffffff; stroke: #d8dbe1; stroke-width: 1.2; }}
    .grid-line {{ fill: none; stroke: #c9ced6; stroke-width: 1.1; }}
    .axis-line {{ fill: none; stroke: #8f96a3; stroke-width: 1.1; opacity: 0.7; }}
    .boundary {{ fill: none; stroke: #2f6fdd; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }}
    .arrow {{ fill: none; stroke: #1b1d24; stroke-width: 2.8; stroke-linecap: round; stroke-linejoin: round; }}
  </style>
  <rect width="1200" height="460" fill="#ffffff" />
  <text x="600" y="38" class="figure-title" text-anchor="middle">The same set under linear and non-linear transformations</text>
  {"".join(panels)}
</svg>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "transformations-shape.svg"
    png_path = OUT_DIR / "transformations-shape.png"
    svg_path.write_text(svg_document(), encoding="utf-8")

    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run(
            [converter, "-w", "1800", "-o", str(png_path), str(svg_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
