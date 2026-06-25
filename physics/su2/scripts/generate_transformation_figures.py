from __future__ import annotations

import math
import shutil
import subprocess
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINEAR_ALGEBRA_OUT_DIR = ROOT / "figures" / "linear-algebra"
LINEAR_TRANSFORMATIONS_OUT_DIR = ROOT / "figures" / "linear-transformations"

Point = tuple[float, float]
Matrix = tuple[tuple[float, float], tuple[float, float]]


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


def matrix_apply(matrix: Matrix, point: Point) -> Point:
    (a, b), (c, d) = matrix
    x, y = point
    return a * x + b * y, c * x + d * y


def determinant(matrix: Matrix) -> float:
    (a, b), (c, d) = matrix
    return a * d - b * c


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


def identity_matrix_svg_document() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 360" role="img" aria-labelledby="identity-title identity-desc">
  <title id="identity-title">The identity matrix leaves a shape unchanged</title>
  <desc id="identity-desc">Two panels show the same grid and arrow-shaped region before and after applying the two-dimensional identity matrix.</desc>
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#4b5565"/>
    </marker>
    <g id="identity-grid">
      <rect x="0" y="0" width="280" height="240" rx="8" fill="#ffffff" stroke="#d6dde8" stroke-width="1.5"/>
      <g stroke="#d8dee8" stroke-width="1">
        <line x1="56" y1="0" x2="56" y2="240"/>
        <line x1="112" y1="0" x2="112" y2="240"/>
        <line x1="168" y1="0" x2="168" y2="240"/>
        <line x1="224" y1="0" x2="224" y2="240"/>
        <line x1="0" y1="48" x2="280" y2="48"/>
        <line x1="0" y1="96" x2="280" y2="96"/>
        <line x1="0" y1="144" x2="280" y2="144"/>
        <line x1="0" y1="192" x2="280" y2="192"/>
      </g>
      <g stroke="#9aa4b2" stroke-width="2">
        <line x1="140" y1="0" x2="140" y2="240"/>
        <line x1="0" y1="120" x2="280" y2="120"/>
      </g>
    </g>
    <polygon id="identity-arrow" points="30,98 156,98 156,70 245,120 156,170 156,142 30,142"/>
  </defs>

  <rect width="960" height="360" rx="8" fill="#f7f9fc" stroke="#d6dde8"/>

  <g transform="translate(80 70)">
    <text x="0" y="-22" fill="#111827" font-size="22" font-weight="700" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">Before</text>
    <use href="#identity-grid"/>
    <use href="#identity-arrow" fill="rgba(107,114,128,0.14)" stroke="#6b7280" stroke-width="3" stroke-linejoin="round"/>
  </g>

  <g transform="translate(600 70)">
    <text x="0" y="-22" fill="#111827" font-size="22" font-weight="700" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">After applying I<tspan baseline-shift="sub" font-size="14">2</tspan></text>
    <use href="#identity-grid"/>
    <use href="#identity-arrow" fill="rgba(124,58,237,0.18)" stroke="#7c3aed" stroke-width="3" stroke-linejoin="round"/>
  </g>

  <text x="480" y="190" text-anchor="middle" fill="#4b5565" font-size="26" font-weight="700" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">same</text>
  <path d="M380 180 H565" fill="none" stroke="#4b5565" stroke-width="2.5" marker-end="url(#arrowhead)"/>
</svg>
"""


def determinant_project(point: Point, x0: float, y0: float, size: float) -> Point:
    min_world = -2.0
    max_world = 2.0
    scale = size / (max_world - min_world)
    x, y = point
    return x0 + (x - min_world) * scale, y0 + (max_world - y) * scale


def determinant_points_attr(points: list[Point], x0: float, y0: float, size: float) -> str:
    return " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (determinant_project(point, x0, y0, size) for point in points)
    )


def determinant_line(
    start: Point,
    end: Point,
    x0: float,
    y0: float,
    size: float,
    class_name: str,
    marker: str = "",
) -> str:
    x1, y1 = determinant_project(start, x0, y0, size)
    x2, y2 = determinant_project(end, x0, y0, size)
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'class="{class_name}"{marker_attr}/>'
    )


def determinant_label(
    point: Point,
    x0: float,
    y0: float,
    size: float,
    text: str,
    class_name: str,
    dx: float = 8.0,
    dy: float = -8.0,
) -> str:
    x, y = determinant_project(point, x0, y0, size)
    return f'<text x="{x + dx:.2f}" y="{y + dy:.2f}" class="{class_name}">{escape(text)}</text>'


def fmt_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def matrix_label(matrix: Matrix) -> str:
    (a, b), (c, d) = matrix
    return f"A = [[{fmt_number(a)}, {fmt_number(b)}], [{fmt_number(c)}, {fmt_number(d)}]]"


def determinant_panel(
    x0: float,
    y0: float,
    title: str,
    matrix: Matrix,
    note: str,
    det_note: str,
) -> str:
    size = 225.0
    plot_x = x0 + 24.0
    plot_y = y0 + 66.0
    text_x = x0 + 275.0
    det_value = determinant(matrix)
    origin = (0.0, 0.0)
    u = matrix_apply(matrix, (1.0, 0.0))
    v = matrix_apply(matrix, (0.0, 1.0))
    parallelogram = [origin, u, (u[0] + v[0], u[1] + v[1]), v]
    title_text = escape(title)
    matrix_text = escape(matrix_label(matrix))
    note_text = escape(note)
    det_note_text = escape(det_note)

    parts = [
        f'<g transform="translate({x0:.1f} {y0:.1f})">',
        '<rect x="0" y="0" width="520" height="285" rx="8" class="det-card"/>',
        f'<text x="22" y="34" class="det-panel-title">{title_text}</text>',
        "</g>",
        f'<g transform="translate({plot_x:.1f} {plot_y:.1f})">',
        f'<rect x="0" y="0" width="{size:.1f}" height="{size:.1f}" rx="5" class="det-frame"/>',
    ]

    for tick in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        parts.append(determinant_line((tick, -2.0), (tick, 2.0), 0, 0, size, "det-grid"))
        parts.append(determinant_line((-2.0, tick), (2.0, tick), 0, 0, size, "det-grid"))

    parts.extend(
        [
            determinant_line((-2.0, 0.0), (2.0, 0.0), 0, 0, size, "det-axis"),
            determinant_line((0.0, -2.0), (0.0, 2.0), 0, 0, size, "det-axis"),
            f'<polygon points="{determinant_points_attr([(0, 0), (1, 0), (1, 1), (0, 1)], 0, 0, size)}" '
            'class="det-unit-square"/>',
            f'<polygon points="{determinant_points_attr(parallelogram, 0, 0, size)}" '
            'class="det-parallelogram"/>',
            determinant_line(origin, u, 0, 0, size, "det-u", "det-arrow-u"),
            determinant_line(origin, v, 0, 0, size, "det-v", "det-arrow-v"),
            determinant_label(u, 0, 0, size, "Ae1", "det-u-label"),
            determinant_label(v, 0, 0, size, "Ae2", "det-v-label", dx=8, dy=16),
            "</g>",
            f'<text x="{text_x:.1f}" y="{y0 + 82:.1f}" class="det-matrix">{matrix_text}</text>',
            f'<text x="{text_x:.1f}" y="{y0 + 116:.1f}" class="det-summary">det A = {det_value:.1f}</text>',
            f'<text x="{text_x:.1f}" y="{y0 + 148:.1f}" class="det-summary">area = |det A| = {abs(det_value):.1f}</text>',
            f'<text x="{text_x:.1f}" y="{y0 + 186:.1f}" class="det-note">{note_text}</text>',
            f'<text x="{text_x:.1f}" y="{y0 + 214:.1f}" class="det-note">{det_note_text}</text>',
        ]
    )

    if abs(det_value) < 1e-9:
        end = (u[0] + v[0], u[1] + v[1])
        parts.append(
            f'<g transform="translate({plot_x:.1f} {plot_y:.1f})">'
            f'{determinant_line(origin, end, 0, 0, size, "det-collapse")}</g>'
        )

    return "\n".join(parts)


def determinant_svg_document() -> str:
    panels = [
        determinant_panel(
            45,
            105,
            "Columns span a parallelogram",
            ((1.2, 0.4), (0.3, 1.1)),
            "The unit square maps to this parallelogram.",
            "Signed area is ad - bc.",
        ),
        determinant_panel(
            635,
            105,
            "det A = 1",
            ((1.0, 0.8), (0.0, 1.0)),
            "Area and orientation are preserved.",
            "This shear is not a rotation.",
        ),
        determinant_panel(
            45,
            425,
            "det A = 0",
            ((1.0, 0.8), (0.0, 0.0)),
            "The square collapses to a line.",
            "The map is not invertible.",
        ),
        determinant_panel(
            635,
            425,
            "det A < 0",
            ((1.0, 0.0), (0.0, -1.0)),
            "Signed area is negative.",
            "Orientation is reversed.",
        ),
    ]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 760" role="img" aria-labelledby="det-title det-desc">
  <title id="det-title">Determinant as the signed area of a transformed unit square</title>
  <desc id="det-desc">Four panels show that a matrix maps the unit square to a parallelogram whose signed area is the determinant, including determinant one, zero, and negative cases.</desc>
  <defs>
    <marker id="det-arrow-u" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#0f9f8f"/>
    </marker>
    <marker id="det-arrow-v" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#c45a11"/>
    </marker>
  </defs>
  <style>
    .det-title {{ font: 700 28px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .det-subtitle {{ font: 500 16px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5565; }}
    .det-card {{ fill: #ffffff; stroke: #d6dde8; stroke-width: 1.4; }}
    .det-panel-title {{ font: 700 20px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .det-frame {{ fill: #ffffff; stroke: #d6dde8; stroke-width: 1.2; }}
    .det-grid {{ stroke: #dde3ec; stroke-width: 1; }}
    .det-axis {{ stroke: #8994a5; stroke-width: 1.7; }}
    .det-unit-square {{ fill: #6b7280; fill-opacity: 0.08; stroke: #6b7280; stroke-width: 1.8; stroke-dasharray: 5 5; }}
    .det-parallelogram {{ fill: #7c3aed; fill-opacity: 0.18; stroke: #7c3aed; stroke-width: 3; stroke-linejoin: round; }}
    .det-u {{ stroke: #0f9f8f; stroke-width: 4; stroke-linecap: round; }}
    .det-v {{ stroke: #c45a11; stroke-width: 4; stroke-linecap: round; }}
    .det-collapse {{ stroke: #7c3aed; stroke-width: 5; stroke-linecap: round; opacity: 0.85; }}
    .det-u-label {{ font: 700 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f9f8f; }}
    .det-v-label {{ font: 700 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #c45a11; }}
    .det-matrix {{ font: 650 15px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #111827; }}
    .det-summary {{ font: 700 18px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .det-note {{ font: 500 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5565; }}
  </style>
  <rect width="1200" height="760" fill="#f7f9fc"/>
  <text x="600" y="42" class="det-title" text-anchor="middle">determinant = signed area scale</text>
  <text x="600" y="72" class="det-subtitle" text-anchor="middle">The columns Ae1 and Ae2 are the sides of the transformed unit square.</text>
  {"".join(panels)}
</svg>
"""


def main() -> None:
    LINEAR_ALGEBRA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LINEAR_TRANSFORMATIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = LINEAR_ALGEBRA_OUT_DIR / "transformations-shape.svg"
    png_path = LINEAR_ALGEBRA_OUT_DIR / "transformations-shape.png"
    svg_path.write_text(svg_document(), encoding="utf-8")
    (LINEAR_TRANSFORMATIONS_OUT_DIR / "identity-matrix.svg").write_text(
        identity_matrix_svg_document(),
        encoding="utf-8",
    )
    (LINEAR_TRANSFORMATIONS_OUT_DIR / "determinant.svg").write_text(
        determinant_svg_document(),
        encoding="utf-8",
    )

    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run(
            [converter, "-w", "1800", "-o", str(png_path), str(svg_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
