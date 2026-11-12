"""Atomic Hartree-Fock orbitals and symmetry-equivalent sp hybrids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pyscf import scf

from .hf_calculations import ELEMENTS, _SHELLS, create_atom, solve_hf


@dataclass
class AtomicHFOrbitals:
    """Atomic HF solution with its saved canonical and valence-shell orbitals."""

    atomic_number: int
    basis: str
    principal_n: int
    molecule: object
    mo_coefficients: np.ndarray
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    shell_orbitals: Dict[str, np.ndarray]

    @property
    def symbol(self) -> str:
        return ELEMENTS[self.atomic_number][0]

    @property
    def valence_orbitals(self) -> Dict[str, np.ndarray]:
        prefix = str(self.principal_n)
        return {
            label: coefficients
            for label, coefficients in self.shell_orbitals.items()
            if label.startswith(prefix)
        }

    @property
    def spin_orbitals(self) -> List[dict]:
        """Return alpha/beta spin-orbital records for the canonical HF set."""
        records = []
        for orbital_index, occupation in enumerate(self.mo_occupations):
            for spin, occupied in (
                ("alpha", occupation > 0),
                ("beta", occupation > 1),
            ):
                records.append(
                    {
                        "spatial_orbital": orbital_index,
                        "spin": spin,
                        "energy_hartree": float(self.mo_energies[orbital_index]),
                        "occupied": bool(occupied),
                        "coefficients": self.mo_coefficients[:, orbital_index].copy(),
                    }
                )
        return records


def _valence_shell_number(atomic_number: int) -> int:
    remaining = atomic_number
    outermost = 1
    for principal_n, _, capacity in _SHELLS:
        count = min(remaining, capacity)
        if count:
            outermost = max(outermost, principal_n)
        remaining -= count
    return outermost


def _ao_angular_momenta(molecule) -> np.ndarray:
    locations = molecule.ao_loc_nr()
    angular_momenta = np.empty(molecule.nao_nr(), dtype=int)
    for shell_index in range(molecule.nbas):
        start, stop = locations[shell_index : shell_index + 2]
        angular_momenta[start:stop] = molecule.bas_angular(shell_index)
    return angular_momenta


def _orbital_angular_weights(mean_field) -> np.ndarray:
    overlap = mean_field.get_ovlp()
    angular_momenta = _ao_angular_momenta(mean_field.mol)
    weights = np.zeros((mean_field.mo_coeff.shape[1], 4))
    for angular_momentum in range(weights.shape[1]):
        indices = np.flatnonzero(angular_momenta == angular_momentum)
        if indices.size:
            block = overlap[np.ix_(indices, indices)]
            coefficients = mean_field.mo_coeff[indices, :]
            weights[:, angular_momentum] = np.einsum(
                "ik,ij,jk->k", coefficients, block, coefficients
            )
    return weights


def _cartesian_p_orbitals(mean_field, orbital_indices: np.ndarray) -> np.ndarray:
    """Build x/y/z p orbitals with one spherical-average HF radial function."""
    molecule = mean_field.mol
    labels = molecule.ao_labels(fmt=False)
    overlap = mean_field.get_ovlp()
    coefficients = mean_field.mo_coeff[:, orbital_indices]
    channel_matrices = []
    for axis in ("x", "y", "z"):
        indices = np.array(
            [index for index, label in enumerate(labels) if label[3] == axis],
            dtype=int,
        )
        channel = np.zeros_like(overlap)
        channel[np.ix_(indices, indices)] = overlap[np.ix_(indices, indices)]
        channel_matrices.append(coefficients.T @ channel @ coefficients)

    x_values, x_vectors = np.linalg.eigh(channel_matrices[0])
    x_axis = x_vectors[:, int(np.argmax(x_values))]
    _, _, complement = np.linalg.svd(x_axis.reshape(1, 3), full_matrices=True)
    yz_subspace = complement[1:, :].T
    y_values, y_vectors = np.linalg.eigh(
        yz_subspace.T @ channel_matrices[1] @ yz_subspace
    )
    y_axis = yz_subspace @ y_vectors[:, int(np.argmax(y_values))]
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    # Fix arbitrary orbital phases so repeated calculations use consistent axes.
    for axis_index in range(3):
        component = rotation[:, axis_index]
        if component[np.argmax(np.abs(component))] < 0:
            rotation[:, axis_index] *= -1
    aligned = coefficients @ rotation

    # Open-shell ROHF can give the three occupied p components slightly
    # different radial shapes. Averaging those radial coefficients restores
    # equivalent Cartesian p functions without changing the saved canonical
    # spin orbitals.
    axis_indices = [
        np.array(
            [index for index, label in enumerate(labels) if label[3] == axis],
            dtype=int,
        )
        for axis in ("x", "y", "z")
    ]
    radial_coefficients = []
    for axis_index, indices in enumerate(axis_indices):
        radial = aligned[indices, axis_index].copy()
        if radial[np.argmax(np.abs(radial))] < 0:
            radial *= -1
        radial_coefficients.append(radial)
    common_radial = np.mean(radial_coefficients, axis=0)
    radial_overlap = overlap[np.ix_(axis_indices[0], axis_indices[0])]
    norm = float(common_radial @ radial_overlap @ common_radial)
    if norm <= 0:
        raise RuntimeError("The averaged HF p-shell radial function has zero norm.")
    common_radial /= np.sqrt(norm)

    cartesian = np.zeros((molecule.nao_nr(), 3))
    for axis_index, indices in enumerate(axis_indices):
        cartesian[indices, axis_index] = common_radial
    return cartesian


def _extract_shell_orbitals(mean_field, highest_n: int) -> Dict[str, np.ndarray]:
    weights = _orbital_angular_weights(mean_field)
    s_indices = np.flatnonzero(weights[:, 0] > 0.95)
    p_indices = np.flatnonzero(weights[:, 1] > 0.95)
    orbitals = {}
    for principal_n in range(1, highest_n + 1):
        if principal_n > len(s_indices):
            raise RuntimeError(f"Could not identify the {principal_n}s orbital.")
        s_index = int(s_indices[principal_n - 1])
        orbitals[f"{principal_n}s"] = mean_field.mo_coeff[:, s_index].copy()
        if principal_n < 2:
            continue
        start = 3 * (principal_n - 2)
        selected_p = p_indices[start : start + 3]
        if selected_p.size != 3:
            raise RuntimeError(f"Could not identify the {principal_n}p orbital triplet.")
        p_coefficients = _cartesian_p_orbitals(mean_field, selected_p)
        for axis_index, axis in enumerate(("x", "y", "z")):
            orbitals[f"{principal_n}p{axis}"] = p_coefficients[
                :, axis_index
            ].copy()
    return orbitals


def calculate_atomic_hf_orbitals(
    atomic_number: int, basis: str = "cc-pVTZ"
) -> AtomicHFOrbitals:
    """Solve an atom and retain its canonical spin and valence-shell orbitals."""
    mean_field = solve_hf(atomic_number, basis=basis)
    principal_n = _valence_shell_number(atomic_number)
    return AtomicHFOrbitals(
        atomic_number=atomic_number,
        basis=basis,
        principal_n=principal_n,
        molecule=mean_field.mol,
        mo_coefficients=mean_field.mo_coeff.copy(),
        mo_energies=mean_field.mo_energy.copy(),
        mo_occupations=mean_field.mo_occ.copy(),
        shell_orbitals=_extract_shell_orbitals(mean_field, principal_n),
    )


def make_equivalent_hybrids(
    atomic_orbitals: AtomicHFOrbitals,
    hybrid_count: int,
    principal_n: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Build orthonormal sp, sp2, or sp3 hybrids from the atomic HF shell."""
    if hybrid_count not in (2, 3, 4):
        raise ValueError("hybrid_count must be 2 (sp), 3 (sp2), or 4 (sp3).")
    shell = atomic_orbitals.principal_n if principal_n is None else principal_n
    if not isinstance(shell, int) or shell < 2 or shell > atomic_orbitals.principal_n:
        raise ValueError("Hybridization requires an available shell with n >= 2.")
    p_labels = [f"{shell}p{axis}" for axis in ("x", "y", "z")]
    if any(label not in atomic_orbitals.shell_orbitals for label in p_labels):
        raise ValueError("Equivalent sp hybrids require a valence-shell p triplet.")

    if hybrid_count == 2:
        directions = np.array(((1, 0, 0), (-1, 0, 0)), dtype=float)
        hybrid_name = "sp"
    elif hybrid_count == 3:
        angles = 2 * np.pi * np.arange(3) / 3
        directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))
        hybrid_name = "sp2"
    else:
        directions = np.array(
            ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
            dtype=float,
        ) / np.sqrt(3.0)
        hybrid_name = "sp3"

    s_orbital = atomic_orbitals.shell_orbitals[f"{shell}s"]
    p_orbitals = np.column_stack(
        [atomic_orbitals.shell_orbitals[label] for label in p_labels]
    )
    s_weight = 1.0 / hybrid_count
    p_weight = (hybrid_count - 1.0) / hybrid_count
    hybrids = {}
    for index, direction in enumerate(directions, start=1):
        hybrids[f"{hybrid_name} hybrid {index}"] = (
            np.sqrt(s_weight) * s_orbital
            + np.sqrt(p_weight) * (p_orbitals @ direction)
        )

    overlap = atomic_orbitals.molecule.intor("int1e_ovlp")
    matrix = np.column_stack(list(hybrids.values()))
    gram = matrix.T @ overlap @ matrix
    if not np.allclose(gram, np.eye(hybrid_count), atol=1e-8):
        raise RuntimeError("The generated hybrid orbitals are not orthonormal.")
    return hybrids


def save_atomic_orbitals(
    atomic_orbitals: AtomicHFOrbitals, path: str | Path
) -> None:
    """Persist the canonical HF spin-orbital coefficients and their basis key."""
    np.savez_compressed(
        path,
        atomic_number=np.asarray(atomic_orbitals.atomic_number),
        basis=np.asarray(atomic_orbitals.basis),
        mo_coefficients=atomic_orbitals.mo_coefficients,
        mo_energies=atomic_orbitals.mo_energies,
        mo_occupations=atomic_orbitals.mo_occupations,
    )


def load_atomic_orbitals(path: str | Path) -> AtomicHFOrbitals:
    """Load saved HF coefficients and reconstruct the matching atomic basis."""
    with np.load(path, allow_pickle=False) as saved:
        atomic_number = int(saved["atomic_number"])
        basis = str(saved["basis"])
        coefficients = saved["mo_coefficients"].copy()
        energies = saved["mo_energies"].copy()
        occupations = saved["mo_occupations"].copy()
    molecule = create_atom(atomic_number, basis=basis)
    mean_field = scf.RHF(molecule)
    mean_field.mo_coeff = coefficients
    mean_field.mo_energy = energies
    mean_field.mo_occ = occupations
    result = AtomicHFOrbitals(
        atomic_number=atomic_number,
        basis=basis,
        principal_n=_valence_shell_number(atomic_number),
        molecule=molecule,
        mo_coefficients=coefficients,
        mo_energies=energies,
        mo_occupations=occupations,
        shell_orbitals={},
    )
    result.shell_orbitals = _extract_shell_orbitals(
        mean_field, result.principal_n
    )
    return result



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
    atomic_orbitals: AtomicHFOrbitals,
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


def plot_atomic_orbital_3d(
    atomic_orbitals: AtomicHFOrbitals,
    coefficients: np.ndarray,
    label: str,
    grid_points: int = 51,
    extent_bohr: float = 5.0,
    isovalue_fraction: Optional[float] = None,
    render_mode: str = "surface",
    probability_coverage: float = 0.95,
    point_count: int = 30000,
    random_seed: int = 12345,
):
    """Interactive phase-colored isosurface or |psi|^2 probability point cloud.

    `probability_coverage` controls the surface cutoff so that approximately
    this fraction of the density on the displayed *grid* lies in the region
    |psi| >= cutoff. It is not a fraction of the peak amplitude. A legacy
    `isovalue_fraction` can optionally override this for old callers.

    Both modes represent a spatial orbital; spin-up/down spatial plots are
    identical in restricted / restricted-open-shell HF.
    """
    import plotly.graph_objects as go

    if grid_points < 25:
        raise ValueError("grid_points must be at least 25.")
    if not np.isfinite(extent_bohr) or extent_bohr <= 0:
        raise ValueError("extent_bohr must be finite and positive.")
    if render_mode not in ("surface", "points"):
        raise ValueError("render_mode must be 'surface' or 'points'.")
    if not 0 < probability_coverage < 1:
        raise ValueError("probability_coverage must be between zero and one.")
    if isovalue_fraction is not None and not 0 < isovalue_fraction < 1:
        raise ValueError("isovalue_fraction must be between zero and one.")
    if not isinstance(point_count, int) or point_count <= 0:
        raise ValueError("point_count must be a positive integer.")

    axis = np.linspace(-extent_bohr, extent_bohr, grid_points)
    x_grid, y_grid, z_grid = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_coordinates = np.column_stack((
        x_grid.ravel(), y_grid.ravel(), z_grid.ravel()
    ))
    # Use batches to avoid materializing a potentially huge AO matrix.
    orbital_values = np.empty(len(grid_coordinates), dtype=float)
    for start in range(0, len(grid_coordinates), 16384):
        stop = min(start + 16384, len(grid_coordinates))
        ao_values = atomic_orbitals.molecule.eval_gto(
            "GTOval_sph", grid_coordinates[start:stop]
        )
        orbital_values[start:stop] = ao_values @ coefficients
    orbital_values = orbital_values.reshape((grid_points,) * 3)
    maximum = float(np.max(np.abs(orbital_values)))
    if not np.isfinite(maximum) or maximum <= 0:
        raise RuntimeError(f"{label} has no finite amplitude on the plotting grid.")

    symbol = atomic_orbitals.symbol
    figure = go.Figure()
    if render_mode == "surface":
        threshold = (
            maximum * isovalue_fraction
            if isovalue_fraction is not None
            else _surface_level_for_probability(orbital_values, probability_coverage)
        )
        if threshold <= 0:
            raise RuntimeError("Grid resolution cannot resolve a positive surface level.")

        for phase_sign, phase, color in (
            (+1, "positive phase", "#2878b5"),
            (-1, "negative phase", "#d34a4a"),
        ):
            phase_values = phase_sign * orbital_values
            if float(np.max(phase_values)) < threshold:
                continue
            # IMPORTANT: a single Isosurface between `threshold` and `maximum`
            # is *not* the contour at `threshold`. Use a near-zero range to
            # force the displayed contour to the desired low-amplitude level.
            figure.add_trace(go.Isosurface(
                x=grid_coordinates[:, 0],
                y=grid_coordinates[:, 1],
                z=grid_coordinates[:, 2],
                value=phase_values.ravel(),
                isomin=threshold,
                isomax=threshold * (1.0 + 1e-5),
                surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=0.48,
                caps={"x": {"show": False}, "y": {"show": False},
                      "z": {"show": False}},
                name=phase,
                showlegend=True,
            ))
        if isovalue_fraction is None:
            note = f"Surface threshold: approximately {probability_coverage:.0%} of in-box grid probability"
        else:
            note = f"Surface threshold: {isovalue_fraction:.3g} of maximum |psi|"
    else:
        sampled_points, sampled_amplitudes, in_box_probability = _sample_probability_points(
            atomic_orbitals, coefficients, orbital_values,
            extent_bohr, point_count, random_seed,
        )
        for positive, phase, color in (
            (True, "positive phase", "#2878b5"),
            (False, "negative phase", "#d34a4a"),
        ):
            mask = sampled_amplitudes >= 0 if positive else sampled_amplitudes < 0
            if not np.any(mask):
                continue
            phase_points = sampled_points[mask]
            figure.add_trace(go.Scatter3d(
                x=phase_points[:, 0],
                y=phase_points[:, 1],
                z=phase_points[:, 2],
                mode="markers",
                marker={"size": 1.6, "color": color, "opacity": 0.55},
                name=phase,
                showlegend=True,
                hoverinfo="skip",
            ))
        note = (
            f"{point_count:,} samples from |psi|² (trilinear grid approximation); "
            f"estimated in-box probability = {in_box_probability:.2f}"
        )

    figure.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker={"size": 6, "color": "#000000"},
        text=[symbol],
        textposition="top center",
        name="nucleus",
    ))
    figure.update_layout(
        title={"text": (
            f"{symbol} {label} — atomic HF / {atomic_orbitals.basis}"
            f"<br><sup>{note}</sup>"
        )},
        scene={
            "xaxis": {"title": "x (Bohr)", "range": [-extent_bohr, extent_bohr]},
            "yaxis": {"title": "y (Bohr)", "range": [-extent_bohr, extent_bohr]},
            "zaxis": {"title": "z (Bohr)", "range": [-extent_bohr, extent_bohr]},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
    )
    return figure
