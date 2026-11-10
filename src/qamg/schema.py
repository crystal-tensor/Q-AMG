from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DesignSpec(BaseModel):
    family: Literal["drive", "clamp"] = "drive"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    groups: int = Field(default=5, ge=3, le=5)
    scale: float = Field(default=1.0, ge=0.6, le=1.6)
    load: Literal["light", "heavy"] = "light"
    compact: bool = False
    shaft_length_mm: float = Field(default=100, ge=70, le=150)
    opening_mm: float = Field(default=30, ge=15, le=50)
    # This is deliberately evaluated by the geometry oracle, not silently assumed satisfied.
    max_gear_radius_mm: float | None = Field(default=None, ge=12, le=40)
    cost_overrides: dict[str, list[int]] = Field(default_factory=dict)
    source_text: str | None = None


class PlannerSettings(BaseModel):
    backend: Literal["auto", "torch-gpu", "torch-cpu", "qb", "qb-noisy", "company", "classical"] = "auto"
    depth: int = Field(default=2, ge=1, le=6)
    iterations: int = Field(default=16, ge=0, le=150)
    training_shots: int = Field(default=128, ge=16, le=8192)
    shots: int = Field(default=512, ge=32, le=16384)
    mixer: Literal["xy", "x"] = "xy"
    encoding: Literal["onehot", "compact_binary"] = "onehot"
    max_feedback_rounds: int = Field(default=3, ge=1, le=8)
    max_candidates: int = Field(default=8, ge=1, le=64)
    max_qubits: int = Field(default=16, ge=6, le=24)
    timeout_s: float = Field(default=120, gt=0, le=1800)
    decomposition_passes: int = Field(default=2, ge=1, le=8)


class Image3DSettings(BaseModel):
    image_model: Literal["partcrafter", "triposr-geometric"] = "partcrafter"
    resolution: int = Field(default=128, ge=64, le=192)
    desired_parts: int = Field(default=3, ge=2, le=12)
    size_mm: float = Field(default=100, ge=10, le=1000)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    remove_background: bool = True
    foreground_ratio: float = Field(default=.85, ge=.5, le=.95)
    density_threshold: float = Field(default=25, ge=1, le=100)
    device: Literal["auto", "cpu", "gpu"] = "auto"
    denoising_steps: int = Field(default=50, ge=10, le=80)
    part_tokens: Literal[512, 1024] = 1024
    planner: PlannerSettings = Field(default_factory=lambda: PlannerSettings(
        encoding="compact_binary", iterations=6, depth=1, shots=512))


class ImagePlanningContext(BaseModel):
    family: Literal["image_parts"] = "image_parts"
    seed: int
    image_sha256: str
    desired_parts: int
    source: str = "Generated single-view mesh; hidden geometry and structural segmentation are inferred"


class ShowcaseContext(BaseModel):
    family: Literal["showcase"] = "showcase"
    example_id: str
    title: str
    seed: int = 42
    source: str = "Authored procedural concept; QB chooses illustrative layout only; not photo inference or official CAD"


class Option(BaseModel):
    key: str
    label: str
    cost: int
    attributes: dict[str, Any] = Field(default_factory=dict)


class ChoiceGroup(BaseModel):
    key: str
    label: str
    options: list[Option]

    @model_validator(mode="after")
    def unique_options(self) -> ChoiceGroup:
        if len(self.options) < 2 or len({o.key for o in self.options}) != len(self.options):
            raise ValueError("A choice group needs at least two distinct options")
        return self


class PairRule(BaseModel):
    left: tuple[int, int]
    right: tuple[int, int]
    cost: int = 0
    forbidden: bool = False
    reason: str = ""


class UnaryRule(BaseModel):
    group: int
    option: int
    reason: str


class ValidationIssue(BaseModel):
    code: str
    message: str
    parts: list[str] = Field(default_factory=list)
    culprit_choices: list[tuple[int, int]] = Field(default_factory=list)


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue]
    checked_pairs: int = 0
    max_intersection_mm3: float = 0
    checks: dict[str, Any] = Field(default_factory=dict)

