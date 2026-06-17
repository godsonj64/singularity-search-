"""Singularity Search.

A defensible implementation of Origin-Coupled Spectral Diffusion Search.
The production path is classical. The quantum path is a small Qiskit
proof-of-concept for amplitude amplification over finite indexed states.
"""

from singularity_search.classical.ocsd import OCSDConfig, OriginCoupledSpectralDiffusionSearch

__all__ = ["OCSDConfig", "OriginCoupledSpectralDiffusionSearch"]
__version__ = "0.1.0"
