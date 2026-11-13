"""Plotting-only orbital density sampling, independent of PySCF imports.

Kept in a separate lightweight module so molecular plotting and its unit
checks can run without initializing the electronic-structure solver. Algorithms
match the unchanged atomic-orbital plotting implementation.
"""
from __future__ import annotations
import numpy as np

def _surface_level_for_probability(
    orbital_values: np.ndarray, probability_coverage: float
) -> float:
    """Amplitude cutoff containing the chosen fraction of in-box grid density.

    The uniform grid makes the cell-volume factor cancel in this calculation.
    This avoids choosing the contour relative to the very large inner s peak.
    """
    density_desc = np.sort(np.square(orbital_values).ravel())[::-1]
    cumulative_density = np.cumsum(density_desc)
    if not np.isfinite(cumulative_density[-1]) or cumulative_density[-1] <= 0:
        raise ValueError("Orbital probability density is zero or non-finite.")
    cutoff_index = int(np.searchsorted(
        cumulative_density, probability_coverage * cumulative_density[-1]
    ))
    return float(np.sqrt(density_desc[min(cutoff_index, len(density_desc)-1)]))


def _sample_probability_points(
    atomic_orbitals: object,
    coefficients: np.ndarray,
    orbital_values: np.ndarray,
    extent_bohr: float,
    point_count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample a trilinear approximation to |psi|^2, returning exact phase signs.

    Each cell's mass is proportional to the average of its eight corner
    densities. Conditional on a cell, choosing a corner weighted by its
    density and sampling three triangular 1D distributions samples the
    trilinearly interpolated density exactly (within the finite grid).
    The signs are evaluated from the *actual HF orbital* at the samples.
    """
    density = np.square(orbital_values)
    n = density.shape[0]
    spacing = 2.0 * extent_bohr / (n - 1)
    corners = np.array(
        [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        dtype=int,
    )
    cell_mass = np.zeros((n - 1, n - 1, n - 1), dtype=float)
    for dx, dy, dz in corners:
        cell_mass += density[
            dx:dx + n - 1, dy:dy + n - 1, dz:dz + n - 1
        ] / 8.0
    cell_mass = cell_mass.ravel()
    total = cell_mass.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Orbital density does not have positive in-box probability.")

    rng = np.random.default_rng(random_seed)
    chosen_cells = rng.choice(cell_mass.size, size=point_count, p=cell_mass / total)
    ix, iy, iz = np.unravel_index(chosen_cells, (n - 1,) * 3)

    # A mixture over the eight corner basis functions gives the exact
    # trilinear interpolant of density inside each selected cell.
    corner_density = np.column_stack([
        density[ix + dx, iy + dy, iz + dz] for dx, dy, dz in corners
    ])
    cdf = np.cumsum(corner_density, axis=1)
    corner_draw = rng.random(point_count) * cdf[:, -1]
    corner_choice = np.sum(corner_draw[:, None] > cdf, axis=1)
    corner_bits = corners[corner_choice]
    uniforms = rng.random((point_count, 3))
    local = np.where(
        corner_bits == 1,
        np.sqrt(uniforms),
        1.0 - np.sqrt(1.0 - uniforms),
    )
    sampled_points = -extent_bohr + spacing * (
        np.column_stack((ix, iy, iz)) + local
    )

    # Evaluate the actual wavefunction at each plotted coordinate, rather
    # than inferring phase from grid nodes (which can straddle a radial node).
    amplitudes = np.empty(point_count, dtype=float)
    for start in range(0, point_count, 8192):
        stop = min(start + 8192, point_count)
        ao_values = atomic_orbitals.molecule.eval_gto(
            "GTOval_sph", sampled_points[start:stop]
        )
        amplitudes[start:stop] = ao_values @ coefficients
    return sampled_points, amplitudes, float(total * spacing**3)

