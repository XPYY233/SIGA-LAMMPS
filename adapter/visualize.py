"""Structure visualisation, via OVITO's Python module.

Turn a trajectory into something a researcher can look at. Two outputs, because
the image alone does not answer a scientific question:

* a **rendered image** of a chosen frame, and
* **numeric observables** — the radial distribution function and a coordination
  breakdown — which are what a physicist actually reads.

The numbers matter as much as the picture. A snapshot shows that atoms exist; the
RDF shows whether the system is crystalline or melted, which is usually the
question behind the question.

Offscreen rendering was verified on this platform rather than assumed, and it has
one trap worth naming: `render_image` succeeds on an empty scene and writes a
tiny valid PNG of nothing. Adding the pipeline to the scene is what makes the
difference — measured here as 1.5 KB of blank against 91 KB of atoms, so a size
check is a cheap guard against silently rendering nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "VISUALISABLE_SUFFIXES",
    "VisualisationError",
    "VisualisationResult",
    "is_visualisable",
    "render_structure",
]

#: File types OVITO can read as a particle configuration.
VISUALISABLE_SUFFIXES = frozenset({".dump", ".lammpstrj", ".xyz", ".cfg", ".data", ".lmp"})

#: A render below this many bytes is treated as an empty scene. The blank PNG
#: OVITO writes for an empty scene measured 1.5 KB here; a real one measured 91 KB.
_MIN_PLAUSIBLE_PNG_BYTES = 5_000


class VisualisationError(RuntimeError):
    """The structure could not be visualised, with a reason worth showing."""


@dataclass
class VisualisationResult:
    """What a visualisation produced."""

    source: str
    image: Path | None = None
    frames: int = 0
    atoms: int = 0
    observables: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "image": self.image.name if self.image else None,
            "frames": self.frames,
            "atoms": self.atoms,
            "observables": self.observables,
            "warning": self.warning,
        }


def is_visualisable(path: Path | str) -> bool:
    """Whether OVITO is likely to read this file as a structure."""
    return Path(path).suffix.lower() in VISUALISABLE_SUFFIXES


def _require_ovito() -> Any:
    try:
        import ovito  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VisualisationError(
            "未安装 OVITO。安装：.venv/bin/python -m pip install ovito\n"
            "macOS arm64 上还需补一个软链（wheel 打包缺陷）：\n"
            "  cd .venv/lib/python3.12/site-packages/ovito/plugins\n"
            "  ln -sf libospray.3.dylib libospray.3.2.0.dylib"
        ) from exc
    except OSError as exc:  # pragma: no cover - the mispackaged-library case
        raise VisualisationError(
            f"OVITO 无法加载底层库：{exc}\n"
            "macOS arm64 的 wheel 里 soname 对不上，补一个软链即可：\n"
            "  cd .venv/lib/python3.12/site-packages/ovito/plugins\n"
            "  ln -sf libospray.3.dylib libospray.3.2.0.dylib"
        ) from exc


def render_structure(
    source: Path | str,
    output: Path | str,
    *,
    frame: int | None = None,
    size: tuple[int, int] = (900, 700),
) -> VisualisationResult:
    """Render one frame of a trajectory and compute its key observables.

    Args:
        source: a LAMMPS dump, xyz, cfg or data file.
        output: where to write the PNG.
        frame: which frame to render. Defaults to the last, since a trajectory's
            final state is usually what is being reported.
        size: image size in pixels.

    Returns:
        The result, including the observables.

    Raises:
        VisualisationError: OVITO is unavailable, the file cannot be read, or the
            render produced nothing.
    """
    _require_ovito()

    from ovito.io import import_file
    from ovito.modifiers import CommonNeighborAnalysisModifier, CoordinationAnalysisModifier
    from ovito.vis import TachyonRenderer, Viewport

    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise VisualisationError(f"找不到文件：{source}")
    if not is_visualisable(source):
        raise VisualisationError(
            f"{source.name} 不是可可视化的结构文件"
            f"（支持 {'、'.join(sorted(VISUALISABLE_SUFFIXES))}）"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = import_file(str(source))
        total_frames = pipeline.source.num_frames
        # The last frame by default: a trajectory's endpoint is what gets reported.
        target = total_frames - 1 if frame is None else max(0, min(frame, total_frames - 1))
        data = pipeline.compute(target)
    except Exception as exc:  # noqa: BLE001 - OVITO raises a wide range
        raise VisualisationError(f"OVITO 无法读取 {source.name}：{exc}") from exc

    result = VisualisationResult(
        source=source.name,
        frames=total_frames,
        atoms=data.particles.count,
    )

    # Observables first: they are the scientific content, and rendering is the
    # part most likely to fail on a given platform.
    try:
        pipeline.modifiers.append(CommonNeighborAnalysisModifier())
        pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=3.5))
        analysed = pipeline.compute(target)
        result.observables["fcc_atoms"] = analysed.attributes.get(
            "CommonNeighborAnalysis.counts.FCC"
        )
        table = analysed.tables.get("coordination-rdf")
        if table is not None:
            xy = table.xy()
            if xy:
                peak = max(xy, key=lambda row: row[1])
                result.observables["rdf_points"] = len(xy)
                result.observables["rdf_first_peak_r"] = round(float(peak[0]), 3)
                result.observables["rdf_first_peak_g"] = round(float(peak[1]), 3)
    except Exception as exc:  # noqa: BLE001
        result.warning = f"观测量计算失败：{exc}"

    try:
        # Without this the scene is empty and OVITO still writes a valid, tiny
        # PNG — a success that shows nothing.
        pipeline.add_to_scene()
        viewport = Viewport(type=Viewport.Type.Ortho)
        viewport.zoom_all()
        viewport.render_image(
            filename=str(output), size=size, renderer=TachyonRenderer()
        )
    except Exception as exc:  # noqa: BLE001
        raise VisualisationError(f"渲染失败：{exc}") from exc

    if not output.is_file() or output.stat().st_size < _MIN_PLAUSIBLE_PNG_BYTES:
        size_bytes = output.stat().st_size if output.is_file() else 0
        raise VisualisationError(
            f"渲染输出过小（{size_bytes} 字节），场景可能是空的。"
            "这通常意味着结构文件里没有可绘制的粒子。"
        )

    result.image = output
    return result
