# Q-AMG

Product name: **Q-AMG — Quantum-assisted Mechanical Generation**.

- Quantum runtime, circuits, compiler and serialization use the pinned native
  QuantumBridge SDK. Do not introduce Qiskit, Aer or PennyLane execution.
- `auto` selects the local CUDA/Metal adapter, with native QB CPU fallback.
  Simulators never establish physical quantum acceleration.
- Image inputs use pinned local PartCrafter joint part generation and SAM2.1
  foreground refinement. TripoSR geometric slicing is explicit legacy research.
  Preserve input provenance, inferred geometry, scale assumptions and decisions.
  Keep the visual encoder in fp32: MPS fp16 failed CPU feature comparisons.
  A correct part count and watertight mesh do not establish semantic correctness.
  Reject fragment clouds and inspect actual rendered parts against the image.
  Never replace a failed reconstruction with a fixed example and report success.
- Mesh partitions must pass solid, union and symmetric-difference checks. Keep
  visual explosion separate from sampled paths and continuous-path certification.
- Preserve neural part boundaries. QB chooses assembly display directions in
  the part-aware path; never claim those measurements created the neural parts.
- Preserve historical experiment data, source snapshots and failed runs. Record
  SDK/version/context changes before comparing experiments.
- Never commit credentials, virtual environments, downloaded weights, temporary
  caches or private user uploads. Keep weight URLs, revisions and checksums.
- Start: `./scripts/demo.sh`. Test: `./scripts/test.sh`.
  Image evidence: `uv run python scripts/validate_image_examples.py --output NEW_DIRECTORY`.
- Public viewer assets come from curated examples via `scripts/export_gallery.py`.
  GitHub Pages serves the viewer; image generation and saved boolean merges need
  the explicitly connected local backend.
