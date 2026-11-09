# Third-party notices

Q-AMG uses the native QuantumBridge SDK from Crystal Tensor, pinned in
`vendor/SOURCES.json`. Its Apache-2.0 license and existing attribution notices
are retained in `vendor/quantumbridge/`.

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
