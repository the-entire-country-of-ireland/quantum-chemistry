"""VSEPR-informed molecular geometry seeds (non-quantum utilities)."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem

COVALENT_RADII_A = {1:.31, 2:.28, 3:1.28, 4:.96, 5:.84, 6:.76,
                    7:.71, 8:.66, 9:.57, 10:.58, 11:1.66, 12:1.41,
                    13:1.21, 14:1.11, 15:1.07, 16:1.05, 17:1.02, 18:1.06}
VALENCE_ELECTRONS = {1:1,2:2,3:1,4:2,5:3,6:4,7:5,8:6,9:7,
                     10:8,11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8}
BOHR_PER_ANGSTROM = 1.8897261254578281

@dataclass
class MolecularSeed:
    smiles: str
    rdkit_mol: object
    symbols: list[str]
    xyz_angstrom: np.ndarray
    vsepr_domains: list[int]
    vsepr_lone_pairs: list[int]


def vsepr_environment(atom) -> tuple[int, int]:
    """Approximate steric number from a conventional Lewis-valence model.

    No attempt to infer hypervalent, metallic, radical or resonance-specific
    electronic structures; this is a seed rather than an HF constraint.
    """
    z = atom.GetAtomicNum()
    if z == 1:
        return 1, 0
    valence = VALENCE_ELECTRONS.get(z)
    if valence is None:
        raise ValueError(f"Only Z=1..18 are supported, found Z={z}")
    bond_sum = sum(float(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
    available = valence - atom.GetFormalCharge() - bond_sum
    pairs = int(max(0, round(available / 2)))
    return atom.GetDegree() + pairs, pairs


def target_vsepr_angle_deg(domains: int, lone_pairs: int) -> float | None:
    if domains == 2:
        return 180.
    if domains == 3:
        return 118. if lone_pairs else 120.
    if domains == 4:
        return {0: 109.471220634, 1: 107.0, 2: 104.5}.get(lone_pairs, 109.471220634)
    return None


def geometry_from_smiles(smiles: str, seed: int = 11,
                         relax_vsepr: bool = True) -> MolecularSeed:
    """RDKit ETKDG conformer plus soft covalent-radius and VSEPR-angle priors.

    Works as an INITIAL GUESS for small closed-shell molecules; not a formal
    conformer search. Optimizing all nuclei with ab-initio HF is essential.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in COVALENT_RADII_A:
            raise ValueError("Only elements H-Ar are supported")
        if atom.GetNumRadicalElectrons():
            raise ValueError("This workflow is restricted to closed-shell RHF molecules")
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("ETKDG failed to produce a geometry; provide a custom geometry")
    # ETKDG incorporates valence angle/torsion priors and is the conformer anchor.
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    # Align with explicit size and VSEPR angle priors, without demanding exact ideal geometry.
    bonds = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = COVALENT_RADII_A[mol.GetAtomWithIdx(a).GetAtomicNum()] + COVALENT_RADII_A[mol.GetAtomWithIdx(b).GetAtomicNum()]
        order = float(bond.GetBondTypeAsDouble())
        length *= {1.0:1., 1.5:.95, 2.0:.90, 3.0:.85}.get(order, 1.)
        bonds.append((a, b, length))
    environments = [vsepr_environment(atom) for atom in mol.GetAtoms()]
    angles = []
    for atom in mol.GetAtoms():
        d, lone = environments[atom.GetIdx()]
        theta = target_vsepr_angle_deg(d, lone)
        if theta is None or atom.GetDegree() < 2:
            continue
        nbs = [n.GetIdx() for n in atom.GetNeighbors()]
        for k in range(len(nbs)):
            for j in range(k+1, len(nbs)):
                angles.append((nbs[k], atom.GetIdx(), nbs[j], np.cos(np.deg2rad(theta))))
    if relax_vsepr:
        original = coords.copy()
        def objective(flat):
            p = flat.reshape(-1, 3)
            score = 0.04 * np.sum((p - original)**2)
            for a,b,length in bonds:
                score += 12.0 * (np.linalg.norm(p[a]-p[b])-length)**2
            for a,o,b,cos_target in angles:
                u, v = p[a]-p[o], p[b]-p[o]
                c = (u @ v)/(max(np.linalg.norm(u)*np.linalg.norm(v),1e-12))
                score += 1.2*(c-cos_target)**2
            return float(score)
        fit = minimize(objective, coords.ravel(), method='L-BFGS-B',
                       options={'maxiter':300, 'ftol':1e-11})
        coords = fit.x.reshape(-1,3)
    coords = align_molecular_axes(coords, mol)
    for i, pos in enumerate(coords):
        mol.GetConformer().SetAtomPosition(i, pos)
    return MolecularSeed(smiles, mol,
                         [a.GetSymbol() for a in mol.GetAtoms()], coords,
                         [d for d,_ in environments], [l for _,l in environments])


def align_molecular_axes(xyz_angstrom: np.ndarray, rdkit_mol) -> np.ndarray:
    """Set a chemically meaningful coordinate frame before AO construction.

    Prefer a central heavy-atom bond for x. For planar molecules select y in
    the molecular plane; z is the plane normal. Linear molecules therefore
    lie on x, with perpendicular pi functions labeled py and pz. Chirality
    and all internuclear distances are preserved by proper (det=+1) rotation.
    """
    xyz = np.asarray(xyz_angstrom, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != rdkit_mol.GetNumAtoms():
        raise ValueError('Coordinates must be an N x 3 array ordered like the molecule')
    if len(xyz) < 2:
        return xyz.copy()
    atoms = list(rdkit_mol.GetAtoms())
    heavy = [a.GetIdx() for a in atoms if a.GetAtomicNum() > 1]
    if len(heavy) >= 2:
        heavy_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b)
                       for b in rdkit_mol.GetBonds()
                       if b.GetBeginAtom().GetAtomicNum() > 1
                       and b.GetEndAtom().GetAtomicNum() > 1]
        if heavy_bonds:
            # Prefer triple/double bonds, and then centers with higher degree.
            left, right, _ = max(heavy_bonds, key=lambda entry: (
                entry[2].GetBondTypeAsDouble(),
                sum(atoms[v].GetDegree() for v in entry[:2]),
                -min(entry[:2])))
        else:
            left, right = heavy[:2]
    elif heavy:
        left = heavy[0]
        neighbors = [a.GetIdx() for a in atoms[left].GetNeighbors()]
        right = neighbors[0] if neighbors else (0 if left != 0 else 1)
    else:
        left, right = 0, 1
    x = xyz[right] - xyz[left]
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        raise ValueError('Bond used for axis alignment has zero length')
    x /= norm
    # Use a second bond direction to put the main molecular plane in xy.
    # Nonplanar molecules still get a deterministic frame, but p labels are
    # only directions, not quantum numbers of the molecular orbitals.
    offsets = xyz - xyz[left]
    y_candidates = offsets - np.outer(offsets @ x, x)
    residuals = np.linalg.norm(y_candidates, axis=1)
    residuals[[left, right]] = 0.0
    idx = int(np.argmax(residuals))
    if residuals[idx] > 1e-7:
        y = y_candidates[idx] / residuals[idx]
    else:
        anchor = np.eye(3)[int(np.argmin(np.abs(x)))]
        y = anchor - x * np.dot(anchor, x)
        y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    rotation = np.column_stack((x, y, z))
    center = xyz.mean(axis=0)
    return (xyz - center) @ rotation
