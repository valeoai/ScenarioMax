"""
Stage 3: Format - Convert unified scenarios to target formats.

This module provides format converters for transforming unified scenarios
into various simulator formats (Waymax, GPUDrive, PufferDrive, etc.).

Submodules:
- waymax: Waymax/V-Max simulator format converter (TFRecord)
- gpudrive: GPUDrive simulator format converter (JSON)
- pufferdrive: PufferDrive simulator format converter (Binary)
"""

__all__ = ["waymax", "gpudrive", "pufferdrive"]
