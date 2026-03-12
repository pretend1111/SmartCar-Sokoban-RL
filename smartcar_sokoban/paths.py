"""Shared project paths."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ASSETS_ROOT = PROJECT_ROOT / "assets"
MAPS_ROOT = ASSETS_ROOT / "maps"
IMAGES_ROOT = ASSETS_ROOT / "images"
IMAGE_CLASS_ROOT = IMAGES_ROOT / "class"
IMAGE_NUM_ROOT = IMAGES_ROOT / "num"
DOCS_ROOT = PROJECT_ROOT / "docs"
RUNS_ROOT = PROJECT_ROOT / "runs"
