from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import DesignSpec, Image3DSettings, PlannerSettings


def main():
    parser = argparse.ArgumentParser(prog="qamg", description="Quantum/classical mechanical design laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report runtime, simulator device and package versions")
    gen = sub.add_parser("generate", help="Generate a validated assembly and measurement trace")
    gen.add_argument("--spec", type=Path)
    gen.add_argument("--text")
    gen.add_argument("--family", choices=["drive", "clamp"], default="drive")
    gen.add_argument("--output", type=Path, required=True)
    for p in [gen]:
        p.add_argument("--backend", choices=["auto", "torch-gpu", "torch-cpu", "qb", "qb-noisy", "company", "classical"], default="auto")
        p.add_argument("--shots", type=int, default=512)
        p.add_argument("--iterations", type=int, default=16)
        p.add_argument("--encoding", choices=["onehot", "compact_binary"], default="onehot")
    examples = sub.add_parser("examples", help="Build both example assemblies using measured circuits")
    examples.add_argument("--output", type=Path, default=Path("artifacts/examples"))
    image = sub.add_parser("image", help="Generate independent neural parts with QB-selected assembly directions")
    image.add_argument("--input", type=Path, required=True)
    image.add_argument("--output", type=Path, required=True)
    image.add_argument("--resolution", type=int, default=128)
    image.add_argument("--parts", type=int, default=3)
    image.add_argument("--size-mm", type=float, default=100)
    image.add_argument("--seed", type=int, default=42)
    image.add_argument("--keep-background", action="store_true")
    image.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    image.add_argument("--backend", choices=["auto", "qb", "qb-noisy", "company", "classical"], default="auto")
    image.add_argument("--image-model", choices=["partcrafter", "triposr-geometric"], default="partcrafter")
    image.add_argument("--steps", type=int, default=50)
    image.add_argument("--tokens", type=int, choices=[512, 1024], default=1024)
    bench = sub.add_parser("benchmark", help="Run preregistered held-out solver experiments")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--instances", type=int, default=30)
    bench.add_argument("--seeds", type=int, default=5)
    bench.add_argument("--iterations", type=int, default=12)
    bench.add_argument("--depth", type=int, default=2)
    bench.add_argument("--shots", type=int, default=256)
    bench.add_argument("--backend", choices=["auto", "torch-gpu", "torch-cpu", "qb", "qb-noisy", "company"], default="qb")
    bench.add_argument("--methods", nargs="+")
    bench.add_argument("--no-geometry", action="store_true")
    bench.add_argument("--split", choices=["train", "heldout"], default="heldout")
    bench.add_argument("--corpus", type=Path)
    bench.add_argument("--warm-start-file", type=Path)
    train = sub.add_parser("train-prior", help="Fit context angles using training split only")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--depth", type=int, default=1)
    serve = sub.add_parser("serve", help="Serve local design and experiment viewer")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    if args.command == "doctor":
        from .runtime import environment_report
        result = environment_report()
    elif args.command == "generate":
        from .pipeline import generate
        if args.spec and args.text:
            parser.error("Choose either --spec or --text")
        if args.spec:
            spec = DesignSpec.model_validate_json(args.spec.read_text())
        elif args.text:
            from .frontends import parse_text
            spec, warnings = parse_text(args.text)
            for message in warnings:
                print(message)
        else:
            spec = DesignSpec(family=args.family)
        result = generate(spec, PlannerSettings(backend=args.backend, shots=args.shots, encoding=args.encoding,
                                               iterations=args.iterations), args.output)
    elif args.command == "examples":
        from .pipeline import generate
        result = {family: generate(DesignSpec(family=family, load="heavy" if family == "drive" else "light"),
                                   PlannerSettings(encoding="compact_binary", iterations=8, shots=256), args.output/family)
                  for family in ["drive", "clamp"]}
    elif args.command == "image":
        from .image_pipeline import generate_image
        settings = Image3DSettings(resolution=args.resolution, desired_parts=args.parts,
                                    image_model=args.image_model, denoising_steps=args.steps, part_tokens=args.tokens,
                                    size_mm=args.size_mm, seed=args.seed, device=args.device,
                                    remove_background=not args.keep_background,
                                    planner=PlannerSettings(backend=args.backend, encoding="compact_binary",
                                                            depth=1, iterations=6, shots=512))
        result = generate_image(args.input.read_bytes(), settings, args.output, args.input.name,
                                lambda message: print(message, flush=True))
    elif args.command == "benchmark":
        if args.instances < 1 or args.seeds < 1:
            parser.error("instances and seeds must be positive")
        from .benchmark import run_benchmark
        result = run_benchmark(args.output,
                               PlannerSettings(backend=args.backend, iterations=args.iterations,
                                               shots=args.shots, depth=args.depth),
                               args.instances, args.seeds, not args.no_geometry,
                               args.corpus, args.methods, args.split, args.warm_start_file)
    elif args.command == "train-prior":
        from .benchmark import train_transfer
        result = train_transfer(args.output, args.corpus, args.depth)
    elif args.command == "serve":
        import uvicorn

        from .server import create_app
        uvicorn.run(create_app(args.output), host=args.host, port=args.port)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

