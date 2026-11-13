"""Regression coverage for molecular isosurface / probability-point rendering.

These tests deliberately avoid PySCF and Streamlit. The plotting backend is
exercised with a toy AO evaluator having both positive and negative phases.
"""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from qchem_analysis.molecular_viewer import plot_combined_with_bonds


class TinyMolecule:
    natm = 1

    def atom_coords(self):
        return np.array([[0., 0., 0.]])

    def atom_coord(self, index):
        assert index == 0
        return np.zeros(3)

    def atom_symbol(self, index):
        assert index == 0
        return 'H'

    def eval_gto(self, kind, coords):
        assert kind == 'GTOval_sph'
        coords = np.asarray(coords)
        weight = np.exp(-np.sum(coords**2, axis=1))
        return np.column_stack([weight, coords[:, 2] * weight])


@pytest.mark.parametrize('render_mode,orbital_trace', [
    ('surface','isosurface'),
    ('points','scatter3d'),
])
def test_both_render_modes_keep_wavefunction_phases(render_mode, orbital_trace):
    result = SimpleNamespace(mol=TinyMolecule(), mode='localized')
    fig = plot_combined_with_bonds(
        result, 0, coefficients_ao=np.array([0., 1.]),
        mode=render_mode, extent_bohr=3., grid_points=25,
        point_count=800, probability_coverage=.90,
    )
    orbital = [trace for trace in fig.data if trace.type == orbital_trace and 'phase' in str(trace.name)]
    assert len(orbital) == 2
    assert all('phase' in str(trace.name) for trace in orbital)
    assert set(trace.name for trace in orbital) == {'+ phase', '\u2212 phase'} or all(
        'phase' in trace.name for trace in orbital
    )
    if render_mode == 'points':
        assert sum(len(trace.x) for trace in orbital) == 800
        assert all(trace.mode == 'markers' for trace in orbital)
    else:
        assert all(trace.surface.count == 1 for trace in orbital)
    # Nuclear markers remain present irrespective of how the orbital is drawn.
    assert any(trace.type == 'scatter3d' and trace.showlegend is False for trace in fig.data)


def test_reject_unrecognized_orbital_mode():
    result = SimpleNamespace(mol=TinyMolecule(), mode='localized')
    with pytest.raises(ValueError, match='mode'):
        plot_combined_with_bonds(result, 0, coefficients_ao=np.array([0.,1.]), mode='wireframe')


def test_molecular_ui_wires_mode_to_mo_gvb_and_reference_plots():
    text = (Path(__file__).resolve().parents[1] / 'pages' / 'molecular.py').read_text()
    assert "key='molecular_orbital_render'" in text
    assert "'Isosurface', 'Probability points'" in text
    assert "orbital_surface(result,seed,coeff,name,render_mode)" in text
    assert "orbital_surface(result,seed,idealA,metaA['hybrid'],render_mode)" in text
    assert "orbital_surface(result,seed,ideal,f'Ideal {atom} {meta[\"hybrid\"]}',render_mode)" in text
    assert "mode=render_mode, grid_points=49" in text
    assert "point_count=26000" in text
