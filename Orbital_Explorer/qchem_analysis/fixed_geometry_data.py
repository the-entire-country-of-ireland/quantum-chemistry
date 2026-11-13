"""Fixed reference geometries and custom XYZ loading; no runtime optimization.

Can be used and tested without PySCF or Streamlit.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from rdkit import Chem
from .molecular_geometry import MolecularSeed, align_molecular_axes, vsepr_environment

_GEOMETRIES=Path(__file__).resolve().parent.parent/'data'/'fixed_geometries.json'

@lru_cache(maxsize=1)
def fixed_molecule_presets() -> dict:
    with _GEOMETRIES.open(encoding='utf-8') as fh:
        return json.load(fh)


def _molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('SMILES is invalid.')
    mol = Chem.AddHs(mol)
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('Open-shell/radical molecules are not supported by this closed-shell model.')
    return mol


def parse_xyz(text: str, symbols: list[str]) -> np.ndarray:
    """Parse plain N x 4 element/x/y/z or conventional XYZ with header.

    Coordinates are matched to RDKit's explicit-H atom order; no silent reorder.
    """
    raw=[line.strip() for line in text.splitlines()]
    lines=[line for line in raw if line]
    if raw and raw[0].isdigit():
        count=int(raw[0])
        if count != len(symbols):
            raise ValueError(f'XYZ header has {count} atoms; SMILES has {len(symbols)}.')
        # Second XYZ line is a freeform comment, even when it has four words.
        lines=[line for line in raw[2:] if line]
    if len(lines)!=len(symbols):
        raise ValueError(f'Expected {len(symbols)} XYZ rows, found {len(lines)}.')
    rows=[]
    for i,(line,symbol) in enumerate(zip(lines,symbols)):
        terms=line.split()
        if len(terms)!=4 or terms[0].lower()!=symbol.lower():
            raise ValueError(f'XYZ row {i+1} must read: {symbol} x y z (SMILES atom order).')
        try:
            rows.append([float(x) for x in terms[1:]])
        except ValueError as exc:
            raise ValueError('XYZ coordinates must be numbers.') from exc
    coords=np.asarray(rows,dtype=float)
    if not np.all(np.isfinite(coords)):
        raise ValueError('XYZ contains non-finite coordinates.')
    if len(coords)>1:
        sep=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=2)
        np.fill_diagonal(sep,np.inf)
        if float(np.min(sep))<.20:
            raise ValueError('Two atoms are closer than 0.20 Å; check XYZ atom order.')
    return coords


def fixed_seed(preset: str, *, smiles: str | None = None,
               xyz_text: str | None = None) -> tuple[MolecularSeed,str]:
    """Built-in precomputed coordinates, or literal supplied XYZ for custom SMILES.

    Custom molecules *require* explicit coordinates. No ETKDG optimization or
    invented equilibrium geometries during the user's calculation.
    """
    if preset=='Custom SMILES + XYZ':
        if not smiles or not smiles.strip():
            raise ValueError('Supply the custom SMILES connectivity.')
        mol=_molecule(smiles.strip())
        symbols=[a.GetSymbol() for a in mol.GetAtoms()]
        if not xyz_text or not xyz_text.strip():
            raise ValueError('Custom molecules require XYZ coordinates; no geometry optimization is performed.')
        xyz=parse_xyz(xyz_text,symbols)
        origin='user-supplied fixed XYZ geometry'
        label=smiles.strip()
    else:
        entry=fixed_molecule_presets().get(preset)
        if entry is None:
            raise ValueError(f'No predefined molecule named {preset!r}')
        label=entry['smiles']
        mol=_molecule(label)
        symbols=[a.GetSymbol() for a in mol.GetAtoms()]
        if symbols!=entry['symbols']:
            raise ValueError(f'Bundled coordinates do not match SMILES atom ordering for {preset}.')
        xyz=np.asarray(entry['coordinates_angstrom'],dtype=float)
        origin=entry['source']
    aligned=align_molecular_axes(xyz,mol)
    conf=mol.GetConformer() if mol.GetNumConformers() else Chem.Conformer(mol.GetNumAtoms())
    for i,pos in enumerate(aligned):
        conf.SetAtomPosition(i,pos)
    if not mol.GetNumConformers():mol.AddConformer(conf,assignId=True)
    env=[vsepr_environment(a) for a in mol.GetAtoms()]
    seed=MolecularSeed(label,mol,symbols,aligned,[v[0] for v in env],[v[1] for v in env])
    return seed,origin



def bond_angles(seed:MolecularSeed, xyz_angstrom:np.ndarray)->list[dict]:
    """Calculated geometric bond angles; not a geometry optimization."""
    from itertools import combinations
    xyz=np.asarray(xyz_angstrom,float)
    result=[]
    for center in seed.rdkit_mol.GetAtoms():
        b=center.GetIdx()
        nbrs=sorted(atom.GetIdx() for atom in center.GetNeighbors())
        for a,c in combinations(nbrs,2):
            v=xyz[a]-xyz[b];w=xyz[c]-xyz[b]
            cosine=float(np.clip(v@w/(np.linalg.norm(v)*np.linalg.norm(w)),-1.,1.))
            result.append({'Angle':f'{seed.symbols[a]}{a+1}–{seed.symbols[b]}{b+1}–{seed.symbols[c]}{c+1}',
                           'Degrees (°)':round(float(np.degrees(np.arccos(cosine))),3)})
    return result
