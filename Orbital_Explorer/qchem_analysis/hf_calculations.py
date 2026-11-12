"""Hartree-Fock and hydrogenic radial-distribution calculations."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pyscf import gto, scf
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import eval_genlaguerre, factorial


ELEMENTS = {
    1: ("H", 1),
    2: ("He", 0),
    3: ("Li", 1),
    4: ("Be", 0),
    5: ("B", 1),
    6: ("C", 2),
    7: ("N", 3),
    8: ("O", 2),
    9: ("F", 1),
    10: ("Ne", 0),
    11: ("Na", 1),
    12: ("Mg", 0),
    13: ("Al", 1),
    14: ("Si", 2),
    15: ("P", 3),
    16: ("S", 2),
    17: ("Cl", 1),
    18: ("Ar", 0),
}

_SHELLS = (
    (1, "s", 2),
    (2, "s", 2),
    (2, "p", 6),
    (3, "s", 2),
    (3, "p", 6),
)
_ANGULAR_MOMENTUM = {0: "s", 1: "p", 2: "d", 3: "f"}


def _validate_atomic_number(z: int) -> None:
    if not isinstance(z, int) or isinstance(z, bool) or z not in ELEMENTS:
        raise ValueError("Atomic number must be an integer from 1 through 18.")


def create_atom(z: int, basis: str = "cc-pVTZ"):
    """Create a neutral atom with its ground-state spin for H through Ar."""
    _validate_atomic_number(z)
    symbol, spin = ELEMENTS[z]
    return gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        unit="Bohr",
        charge=0,
        spin=spin,
        symmetry=False,
        verbose=0,
    )


def solve_hf(z: int, basis: str = "cc-pVTZ"):
    """Run a restricted or restricted-open-shell Hartree-Fock calculation."""
    mol = create_atom(z, basis=basis)
    mean_field = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 100
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"Hartree-Fock did not converge for {ELEMENTS[z][0]}.")
    return mean_field


def calculate_hydrogenic_rdf(
    n: int, l: int, z_eff: float, r: np.ndarray
) -> np.ndarray:
    """Return the normalized hydrogen-like radial probability density."""
    if not isinstance(n, int) or not isinstance(l, int) or n < 1 or l < 0 or l >= n:
        raise ValueError("Quantum numbers must satisfy n >= 1 and 0 <= l < n.")
    if z_eff <= 0:
        raise ValueError("Effective nuclear charge must be positive.")

    radius = np.asarray(r, dtype=float)
    rho = 2.0 * z_eff * radius / n
    normalization = np.sqrt(
        (2.0 * z_eff / n) ** 3
        * factorial(n - l - 1, exact=False)
        / (2.0 * n * factorial(n + l, exact=False))
    )
    radial_wavefunction = (
        normalization
        * np.exp(-rho / 2.0)
        * rho**l
        * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    )
    return radius**2 * radial_wavefunction**2


def fit_hydrogenic_rdf(
    target_rdf: np.ndarray,
    n: int,
    l: int,
    r: np.ndarray,
    max_z_eff: float,
) -> Tuple[np.ndarray, float]:
    """Fit a hydrogenic RDF's effective charge to a normalized target RDF."""
    radius = np.asarray(r, dtype=float)
    target = np.asarray(target_rdf, dtype=float)
    if radius.ndim != 1 or target.shape != radius.shape or radius.size < 2:
        raise ValueError("target_rdf and r must be one-dimensional arrays of equal length.")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(target)):
        raise ValueError("target_rdf and r must contain only finite values.")
    if np.any(radius < 0) or np.any(np.diff(radius) <= 0):
        raise ValueError("r must be nonnegative and strictly increasing.")
    if np.any(target < 0):
        raise ValueError("target_rdf must be nonnegative.")
    if not np.isfinite(max_z_eff) or max_z_eff <= 0:
        raise ValueError("max_z_eff must be finite and positive.")
    target_integral = trapezoid(target, radius)
    if target_integral <= 0:
        raise ValueError("target_rdf must have a positive integral.")
    target = target / target_integral

    lower_bound = min(0.05, max_z_eff)
    log_bounds = np.log(lower_bound), np.log(max_z_eff)

    def hydrogenic_model(log_z_eff: float) -> np.ndarray:
        return calculate_hydrogenic_rdf(n, l, float(np.exp(log_z_eff)), radius)

    def objective(log_z_eff: float) -> float:
        residual = target - hydrogenic_model(log_z_eff)
        return float(trapezoid(residual**2, radius))

    if log_bounds[0] == log_bounds[1]:
        best_log_z_eff = log_bounds[0]
    else:
        trial_points = np.linspace(*log_bounds, 81)
        trial_values = np.array([objective(point) for point in trial_points])
        best_index = int(np.argmin(trial_values))
        best_log_z_eff = float(trial_points[best_index])
        best_error = float(trial_values[best_index])
        if 0 < best_index < len(trial_points) - 1:
            result = minimize_scalar(
                objective,
                bounds=(trial_points[best_index - 1], trial_points[best_index + 1]),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if not result.success:
                raise RuntimeError(f"Effective-charge fit failed: {result.message}")
            if result.fun < best_error:
                best_log_z_eff = float(result.x)

    z_eff = float(np.exp(best_log_z_eff))
    return hydrogenic_model(best_log_z_eff), z_eff


def effective_nuclear_charge(z: int, n: int, l: int) -> float:
    """Estimate shell screening using Slater's rules for occupied s/p shells."""
    _validate_atomic_number(z)
    if (n, l) not in ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1)):
        raise ValueError("Only occupied 1s, 2s, 2p, 3s, and 3p shells are supported.")

    electron_counts = {}
    remaining = z
    for shell_n, shell_l, capacity in _SHELLS:
        count = min(remaining, capacity)
        electron_counts[(shell_n, shell_l)] = count
        remaining -= count

    target_group = sum(
        count
        for (shell_n, _), count in electron_counts.items()
        if shell_n == n
    )
    shielding = (0.30 if n == 1 else 0.35) * (target_group - 1)

    for (shell_n, _), count in electron_counts.items():
        if shell_n == n:
            continue
        if shell_n == n - 1:
            shielding += 0.85 * count
        elif shell_n < n - 1:
            shielding += count

    return max(float(z) - shielding, 1e-8)


def _angular_momentum_by_ao(mol) -> np.ndarray:
    ao_locations = mol.ao_loc_nr()
    angular_momenta = np.empty(mol.nao_nr(), dtype=int)
    for shell_index in range(mol.nbas):
        start, stop = ao_locations[shell_index : shell_index + 2]
        angular_momenta[start:stop] = mol.bas_angular(shell_index)
    return angular_momenta


def _occupied_orbital_shells(mean_field) -> Dict[str, List[int]]:
    coefficients = mean_field.mo_coeff
    occupied = np.flatnonzero(mean_field.mo_occ > 0)
    ao_l = _angular_momentum_by_ao(mean_field.mol)
    overlap = mean_field.get_ovlp()

    orbital_groups: Dict[int, List[Tuple[float, int]]] = {}
    for orbital_index in occupied:
        coeff = coefficients[:, orbital_index]
        weights = {}
        for angular_momentum in np.unique(ao_l):
            indices = np.flatnonzero(ao_l == angular_momentum)
            weights[angular_momentum] = float(
                coeff[indices] @ overlap[np.ix_(indices, indices)] @ coeff[indices]
            )
        angular_momentum = max(weights, key=weights.get)
        orbital_groups.setdefault(angular_momentum, []).append(
            (float(mean_field.mo_energy[orbital_index]), int(orbital_index))
        )

    shell_orbitals: Dict[str, List[int]] = {}
    for angular_momentum, orbitals in orbital_groups.items():
        orbitals.sort()
        orbitals_per_shell = 2 * angular_momentum + 1
        for radial_index, start in enumerate(range(0, len(orbitals), orbitals_per_shell)):
            principal_number = angular_momentum + 1 + radial_index
            letter = _ANGULAR_MOMENTUM.get(angular_momentum)
            if letter is not None and principal_number <= 3:
                shell_orbitals[f"{principal_number}{letter}"] = [
                    index for _, index in orbitals[start : start + orbitals_per_shell]
                ]
    return shell_orbitals


def extract_rdfs_from_hf(
    mean_field, r: np.ndarray, angular_points: int = 96, radial_chunk_size: int = 16
) -> Dict[str, np.ndarray]:
    """Spherically average occupied HF orbitals and return per-orbital RDFs."""
    radius = np.asarray(r, dtype=float)
    if radius.ndim != 1 or radius.size < 2 or np.any(radius < 0):
        raise ValueError("r must be a one-dimensional array with at least two nonnegative values.")
    if angular_points < 24 or radial_chunk_size < 1:
        raise ValueError("angular_points must be >= 24 and radial_chunk_size must be positive.")

    # Product Gauss-Legendre and uniform-azimuth quadrature integrates over the sphere.
    n_mu = max(6, int(np.sqrt(angular_points / 2)))
    n_phi = max(12, 2 * n_mu)
    mu, mu_weights = np.polynomial.legendre.leggauss(n_mu)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    sin_theta = np.sqrt(1.0 - mu**2)
    directions = np.stack(
        (
            (sin_theta[:, None] * np.cos(phi)).ravel(),
            (sin_theta[:, None] * np.sin(phi)).ravel(),
            np.broadcast_to(mu[:, None], (n_mu, n_phi)).ravel(),
        ),
        axis=1,
    )
    angular_weights = np.repeat(mu_weights, n_phi) * (2.0 * np.pi / n_phi)

    shell_orbitals = _occupied_orbital_shells(mean_field)
    if not shell_orbitals:
        raise RuntimeError("No occupied s or p orbitals were found in the HF solution.")
    orbital_indices = sorted(
        {index for indices in shell_orbitals.values() for index in indices}
    )
    orbital_column = {index: column for column, index in enumerate(orbital_indices)}
    orbital_rdfs = np.zeros((len(orbital_indices), radius.size), dtype=float)

    for start in range(0, radius.size, radial_chunk_size):
        stop = min(start + radial_chunk_size, radius.size)
        radii = radius[start:stop]
        coordinates = (
            radii[:, None, None] * directions[None, :, :]
        ).reshape(-1, 3)
        ao_values = mean_field.mol.eval_gto("GTOval_sph", coordinates)
        mo_values = ao_values @ mean_field.mo_coeff[:, orbital_indices]
        mo_values = mo_values.reshape(len(radii), len(directions), -1)
        angular_density = np.einsum(
            "rak,a->rk", mo_values**2, angular_weights, optimize=True
        )
        orbital_rdfs[:, start:stop] = (
            angular_density.T * radii**2
        )

    results = {}
    for shell, indices in shell_orbitals.items():
        columns = [orbital_column[index] for index in indices]
        rdf = orbital_rdfs[columns].mean(axis=0)
        integral = trapezoid(rdf, radius)
        if not np.isfinite(integral) or integral <= 0:
            raise RuntimeError(f"Could not normalize the HF radial distribution for {shell}.")
        results[shell] = rdf / integral
    return results


def calculate_all_hf_rdfs(
    max_z: int = 18,
    r_max: float = 5.0,
    n_points: int = 3001,
    basis: str = "cc-pVTZ",
) -> Tuple[
    Dict[int, Dict[str, np.ndarray]],
    Dict[int, Dict[str, np.ndarray]],
    Dict[int, Dict[str, float]],
    np.ndarray,
]:
    """Calculate HF and fitted hydrogenic RDFs from H through max_z."""
    if not isinstance(max_z, int) or not 1 <= max_z <= 18:
        raise ValueError("max_z must be an integer from 1 through 18.")
    if r_max <= 0 or n_points < 2:
        raise ValueError("r_max must be positive and n_points must be at least 2.")

    radius = np.linspace(0.0, r_max, n_points)
    hf_results = {}
    hydrogenic_results = {}
    effective_charges = {}

    for z in range(1, max_z + 1):
        print(f"Computing HF for Z={z} ({ELEMENTS[z][0]})...")
        mean_field = solve_hf(z, basis=basis)
        hf_shells = extract_rdfs_from_hf(mean_field, radius)
        hf_results[z] = hf_shells
        hydrogenic_results[z] = {}
        effective_charges[z] = {}
        for shell in hf_shells:
            principal_number = int(shell[:-1])
            angular_momentum = next(
                angular_momentum
                for angular_momentum, letter in _ANGULAR_MOMENTUM.items()
                if letter == shell[-1]
            )
            hydrogenic_results[z][shell], effective_charges[z][shell] = (
                fit_hydrogenic_rdf(
                    hf_shells[shell],
                    principal_number,
                    angular_momentum,
                    radius,
                    float(z),
                )
            )

    return hf_results, hydrogenic_results, effective_charges, radius
