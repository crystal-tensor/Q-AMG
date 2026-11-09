# Third-party notices

Q-AMG uses the native QuantumBridge SDK from Crystal Tensor, pinned in
`vendor/SOURCES.json`. Its Apache-2.0 license and existing attribution notices
are retained in `vendor/quantumbridge/`.

PartCrafter is from the official wgsxm/PartCrafter repository. Its MIT license is
retained in `vendor/partcrafter/LICENSE`; the source revision and local inference
adaptations are recorded in `vendor/SOURCES.json`. The public model revision,
file sizes and SHA-256 hashes are pinned in `configs/partcrafter-files.json`.

SAM2.1 large and its Python inference source are from Meta's official SAM2
project. The Apache-2.0 license and BSD third-party attribution are retained in
`vendor/sam2/`; weights are pinned in `configs/vision-model.json`.

TripoSR source and its example inputs are from VAST-AI-Research / Stability AI /
Tripo AI, pinned in `vendor/SOURCES.json`. Its MIT license is retained in
`vendor/triposr/LICENSE`. Local adaptations are documented in that manifest.
TripoSR model weights use the model publisher's MIT license; the download script
keeps their public revision and SHA-256 rather than storing large weights in Git.

The small U2NetP background model is distributed through the rembg project.
The exact download URL and checksum are recorded in `scripts/prepare_models.py`
and `models/manifest.json`. The local inference path uses ONNX Runtime.

Other Python and JavaScript dependencies remain separate packages, installed
through `uv.lock` and `web/package-lock.json`; their respective licenses apply.

Generated single-image geometry is an estimate. Publication of third-party
example inputs does not establish rights to unrelated uploaded images. Private
uploads are not included in the public gallery or source push.
