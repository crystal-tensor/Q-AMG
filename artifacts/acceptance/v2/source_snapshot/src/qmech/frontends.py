"""Bounded text grammar; image/learned decoding are explicit extension points."""
from __future__ import annotations

import re
from typing import Protocol

from .schema import DesignSpec


class ImageDesignFrontend(Protocol):
    """A future model must expose references + uncertainties, not guess exact hidden CAD."""

    def infer_spec(self, image: bytes) -> tuple[DesignSpec, list[str]]: ...


def parse_text(text: str) -> tuple[DesignSpec, list[str]]:
    if not text.strip():
        raise ValueError("请输入设计要求")
    value = text.lower()
    if any(word in value for word in ["夹", "clamp", "vise", "gripper"]):
        family = "clamp"
    elif any(word in value for word in ["轴", "齿轮", "drive", "gear", "bearing"]):
        family = "drive"
    else:
        raise ValueError("当前语法支持轴—轴承—齿轮装配或夹持机构，请明确选择一类")
    fields = {"family": family, "source_text": text,
              "load": "heavy" if any(w in value for w in ["重载", "heavy"]) else "light",
              "compact": any(w in value for w in ["紧凑", "compact"])}
    for pattern, key in [
        (r"(?:轴长|shaft\s*length)\s*[:：=]?\s*(\d+(?:\.\d+)?)", "shaft_length_mm"),
        (r"(?:开口|opening)\s*[:：=]?\s*(\d+(?:\.\d+)?)", "opening_mm"),
        (r"(?:最大齿轮半径|max\s*gear\s*radius)\s*[:：=]?\s*(\d+(?:\.\d+)?)", "max_gear_radius_mm"),
    ]:
        match = re.search(pattern, value)
        if match:
            fields[key] = float(match.group(1))
    return DesignSpec.model_validate(fields), [
        "当前使用有限设计语法（单位 mm），不是通用语言模型或图片逆向工程。",
        "未列出的要求尚未自动编码，请检查结构化参数和最终校验结果。"]

