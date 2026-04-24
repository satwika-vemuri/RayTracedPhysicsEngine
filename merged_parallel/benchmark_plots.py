#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import mean


def load_run(csv_path: Path) -> dict:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    def avg(col: str) -> float:
        return mean(float(r[col]) for r in rows)

    def first(col: str) -> str:
        return rows[0][col]

    run = {
        "source_csv": str(csv_path),
        "box_dimension": int(first("box_dimension")),
        "frames": len(rows),
        "substeps": int(first("substeps")),
        "write_frames": int(first("write_frames")),
        "bruteforce": int(first("bruteforce")),
        "include_sphere": int(first("include_sphere")),
        "physics_ms": avg("physics_ms"),
        "march_ms": avg("march_ms"),
        "tri_construct_ms": avg("tri_construct_ms"),
        "binning_ms": avg("binning_ms"),
        "flatten_ms": avg("flatten_ms"),
        "h2d_ms": avg("h2d_ms"),
        "ray_ms": avg("ray_ms"),
        "d2h_ms": avg("d2h_ms"),
        "file_ms": avg("file_ms"),
        "total_ms": avg("total_ms"),
        "num_particles": avg("num_particles"),
        "num_triangles": avg("num_triangles"),
        "flat_triangle_indices": avg("flat_triangle_indices"),
        "nonempty_cells": avg("nonempty_cells"),
        "max_tris_per_cell": avg("max_tris_per_cell"),
        "avg_tris_per_nonempty_cell": avg("avg_tris_per_nonempty_cell"),
        "duplication_factor": avg("duplication_factor"),
        "h2d_bytes": avg("h2d_bytes"),
        "d2h_bytes": avg("d2h_bytes"),
        "h2d_bandwidth_gbps": avg("h2d_bandwidth_gbps"),
        "d2h_bandwidth_gbps": avg("d2h_bandwidth_gbps"),
    }

    run["preprocess_cpu_ms"] = (
        run["tri_construct_ms"] + run["binning_ms"] + run["flatten_ms"]
    )
    run["communication_ms"] = run["h2d_ms"] + run["d2h_ms"]
    run["ray_stage_ms"] = run["h2d_ms"] + run["ray_ms"] + run["d2h_ms"]
    run["mesh_stage_ms"] = run["march_ms"]
    run["render_pipeline_ms"] = (
        run["march_ms"]
        + run["tri_construct_ms"]
        + run["binning_ms"]
        + run["flatten_ms"]
        + run["h2d_ms"]
        + run["ray_ms"]
        + run["d2h_ms"]
    )
    run["gpu_compute_ms"] = run["march_ms"] + run["ray_ms"]
    run["other_ms"] = max(
        0.0,
        run["total_ms"]
        - (
            run["physics_ms"]
            + run["preprocess_cpu_ms"]
            + run["gpu_compute_ms"]
            + run["communication_ms"]
            + run["file_ms"]
        ),
    )
    return run


def write_summary_csv(runs: list[dict], out_path: Path) -> None:
    fieldnames = [
        "box_dimension",
        "frames",
        "substeps",
        "write_frames",
        "bruteforce",
        "include_sphere",
        "physics_ms",
        "march_ms",
        "tri_construct_ms",
        "binning_ms",
        "flatten_ms",
        "preprocess_cpu_ms",
        "mesh_stage_ms",
        "ray_stage_ms",
        "render_pipeline_ms",
        "gpu_compute_ms",
        "communication_ms",
        "h2d_ms",
        "ray_ms",
        "d2h_ms",
        "file_ms",
        "other_ms",
        "total_ms",
        "num_particles",
        "num_triangles",
        "flat_triangle_indices",
        "nonempty_cells",
        "max_tris_per_cell",
        "avg_tris_per_nonempty_cell",
        "duplication_factor",
        "h2d_bandwidth_gbps",
        "d2h_bandwidth_gbps",
        "source_csv",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({k: run[k] for k in fieldnames})


def write_markdown_summary(runs: list[dict], out_path: Path) -> None:
    with out_path.open("w") as f:
        f.write("# Box Dimension Benchmark Summary\n\n")
        f.write(
            "| boxDim | total ms/frame | ray stage ms/frame | accel setup ms/frame | "
            "render pipeline ms/frame | communication ms/frame | avg tris/nonempty cell |\n"
        )
        f.write(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for run in runs:
            f.write(
                f"| {run['box_dimension']} | {run['total_ms']:.3f} | {run['ray_stage_ms']:.3f} | "
                f"{run['preprocess_cpu_ms']:.3f} | {run['render_pipeline_ms']:.3f} | "
                f"{run['communication_ms']:.3f} | {run['avg_tris_per_nonempty_cell']:.3f} |\n"
            )

        best_total = min(runs, key=lambda r: r["total_ms"])
        best_ray_stage = min(runs, key=lambda r: r["ray_stage_ms"])
        f.write("\n## Highlights\n\n")
        f.write(
            f"- Lowest total runtime: `boxDim={best_total['box_dimension']}` "
            f"at `{best_total['total_ms']:.3f} ms/frame`\n"
        )
        f.write(
            f"- Lowest end-to-end ray stage time: `boxDim={best_ray_stage['box_dimension']}` "
            f"at `{best_ray_stage['ray_stage_ms']:.3f} ms/frame`\n"
        )


def make_plots(runs: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate plots. Install it or rerun in an environment with matplotlib."
        ) from exc

    boxdims = [r["box_dimension"] for r in runs]

    plt.figure(figsize=(8, 5))
    plt.plot(boxdims, [r["total_ms"] for r in runs], marker="o", linewidth=2)
    plt.xlabel("Box dimension")
    plt.ylabel("Average total runtime (ms/frame)")
    plt.title("Total Runtime vs Box Dimension")
    plt.grid(True, alpha=0.3)
    plt.xticks(boxdims)
    plt.tight_layout()
    plt.savefig(out_dir / "boxdim_total_runtime.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        boxdims,
        [r["ray_stage_ms"] for r in runs],
        marker="o",
        linewidth=2,
        label="ray stage (h2d + kernel + d2h)",
    )
    plt.plot(
        boxdims,
        [r["preprocess_cpu_ms"] for r in runs],
        marker="o",
        linewidth=2,
        label="acceleration setup (tri + bin + flat)",
    )
    plt.plot(
        boxdims,
        [r["mesh_stage_ms"] for r in runs],
        marker="o",
        linewidth=2,
        label="march stage",
    )
    plt.plot(
        boxdims,
        [r["render_pipeline_ms"] for r in runs],
        marker="o",
        linewidth=2,
        label="render pipeline (march + setup + ray stage)",
    )
    plt.xlabel("Box dimension")
    plt.ylabel("Average time (ms/frame)")
    plt.title("Stage Costs vs Box Dimension")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(boxdims)
    plt.tight_layout()
    plt.savefig(out_dir / "boxdim_stage_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        boxdims,
        [r["avg_tris_per_nonempty_cell"] for r in runs],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Box dimension")
    plt.ylabel("Average triangles per non-empty cell")
    plt.title("Cell Occupancy vs Box Dimension")
    plt.grid(True, alpha=0.3)
    plt.xticks(boxdims)
    plt.tight_layout()
    plt.savefig(out_dir / "boxdim_cell_occupancy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    width = 0.7
    x = list(range(len(boxdims)))

    preprocess = [r["preprocess_cpu_ms"] for r in runs]
    mesh = [r["mesh_stage_ms"] for r in runs]
    ray_stage = [r["ray_stage_ms"] for r in runs]
    file_io = [r["file_ms"] for r in runs]
    physics = [r["physics_ms"] for r in runs]

    plt.bar(x, physics, width, label="physics stage")
    plt.bar(x, mesh, width, bottom=physics, label="march stage")

    bottom2 = [a + b for a, b in zip(physics, mesh)]
    plt.bar(x, preprocess, width, bottom=bottom2, label="acceleration setup")

    bottom3 = [a + b for a, b in zip(bottom2, preprocess)]
    plt.bar(x, ray_stage, width, bottom=bottom3, label="ray stage")

    bottom4 = [a + b for a, b in zip(bottom3, ray_stage)]
    plt.bar(x, file_io, width, bottom=bottom4, label="file")

    plt.xticks(x, boxdims)
    plt.xlabel("Box dimension")
    plt.ylabel("Average time (ms/frame)")
    plt.title("End-to-End Runtime Composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "boxdim_runtime_composition.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark CSV files and generate box-dimension plots."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Benchmark CSV files produced by physics_sim",
    )
    parser.add_argument(
        "--outdir",
        default="boxdim_plots",
        help="Directory for plots and aggregated summaries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(Path(p)) for p in args.csv_files]
    runs.sort(key=lambda r: r["box_dimension"])

    write_summary_csv(runs, out_dir / "boxdim_summary.csv")
    write_markdown_summary(runs, out_dir / "boxdim_summary.md")
    make_plots(runs, out_dir)

    print(f"Wrote aggregated summary CSV to {out_dir / 'boxdim_summary.csv'}")
    print(f"Wrote markdown summary to {out_dir / 'boxdim_summary.md'}")
    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
