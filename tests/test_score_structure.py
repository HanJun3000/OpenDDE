# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from opendde.data.inference.input_validation import validate_inference_jobs
from opendde.data.inference.score_structure import (
    build_score_coordinates,
    overlay_score_coordinates,
)
from tests.test_custom_templates import ATOM_OFFSETS, _backbone_atoms, _write_pdb


def _model_atom_array(chains, with_oxt=True) -> AtomArray:
    """Atoms the JSON parser would build: N/CA/C/O(/CB) per residue, OXT at the end."""
    rows = []
    for asym_id, (chain_id, sequence) in enumerate(chains):
        for i, code in enumerate(sequence):
            names = [
                name for name in ATOM_OFFSETS if not (code == "G" and name == "CB")
            ]
            if with_oxt and i == len(sequence) - 1:
                names.append("OXT")
            rows += [(asym_id, chain_id, i + 1, name) for name in names]
    atom_array = AtomArray(len(rows))
    atom_array.add_annotation("asym_id_int", int)
    for i, (asym_id, chain_id, res_id, name) in enumerate(rows):
        atom_array.asym_id_int[i] = asym_id
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = res_id
        atom_array.atom_name[i] = name
    return atom_array


def _job(path, *sequences, chain_ids=None):
    spec = {"structurePath": str(path)}
    if chain_ids is not None:
        spec["chainIds"] = chain_ids
    return {
        "name": "score",
        "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in sequences],
        "scoreStructure": spec,
    }


def test_dimer_coordinates_map_by_chain_and_atom(tmp_path):
    path = tmp_path / "dimer.pdb"
    atoms_a = _backbone_atoms("ACDE", chain_id="A")
    atoms_b = [
        (c, r, n, a, x + 50.0)
        for c, r, n, a, x in _backbone_atoms("GHIK", chain_id="B")
    ]
    _write_pdb(path, atoms_a + atoms_b)
    atom_array = _model_atom_array([("A", "ACDE"), ("B", "GHIK")])

    coords, mask = build_score_coordinates(_job(path, "ACDE", "GHIK"), atom_array)

    # Every atom except the two OXT (absent from the file) is taken from it.
    assert (~mask).sum() == 2
    assert set(atom_array.atom_name[~mask]) == {"OXT"}
    ca_b2 = np.flatnonzero(
        (atom_array.asym_id_int == 1)
        & (atom_array.res_id == 2)
        & (atom_array.atom_name == "CA")
    )[0]
    expected = [c for c in atoms_b if c[1] == 2 and c[3] == "CA"][0][4]
    np.testing.assert_allclose(coords[ca_b2], expected, atol=1e-3)


def test_chain_ids_pick_file_chains_in_model_order(tmp_path):
    path = tmp_path / "dimer.pdb"
    atoms_a = _backbone_atoms("ACDE", chain_id="A")
    atoms_b = [
        (c, r, n, a, x + 50.0)
        for c, r, n, a, x in _backbone_atoms("GHIK", chain_id="B")
    ]
    _write_pdb(path, atoms_a + atoms_b)
    # Model chain order is B's sequence first.
    atom_array = _model_atom_array([("A", "GHIK"), ("B", "ACDE")])

    coords, mask = build_score_coordinates(
        _job(path, "GHIK", "ACDE", chain_ids=["B", "A"]), atom_array
    )

    first_ca = np.flatnonzero(
        (atom_array.asym_id_int == 0) & (atom_array.atom_name == "CA")
    )[0]
    assert mask[first_ca]
    assert coords[first_ca][0] > 40.0  # came from file chain B


def test_mismatched_residue_keeps_backbone_only(tmp_path, monkeypatch):
    path = tmp_path / "mutant.pdb"
    _write_pdb(path, _backbone_atoms("ACDE", chain_id="A"))
    # D3 -> A: not identical, not a substring; map position-wise without Kalign.
    from opendde.data.inference import score_structure

    monkeypatch.setattr(
        score_structure,
        "map_query_to_template",
        lambda q, t, **kwargs: {i: i for i in range(len(q))},
    )
    atom_array = _model_atom_array([("A", "ACAE")])

    _, mask = build_score_coordinates(_job(path, "ACAE"), atom_array)

    res3 = atom_array.res_id == 3
    assert mask[res3 & np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])].all()
    assert not mask[res3 & (atom_array.atom_name == "CB")].any()


def test_chain_count_mismatch_is_reported(tmp_path):
    path = tmp_path / "monomer.pdb"
    _write_pdb(path, _backbone_atoms("ACDE", chain_id="A"))
    atom_array = _model_atom_array([("A", "ACDE"), ("B", "ACDE")])

    with pytest.raises(ValueError, match="2 model chain"):
        build_score_coordinates(_job(path, "ACDE", "ACDE"), atom_array)


def _random_rotation(generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_overlay_full_mask_returns_given_coordinates():
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(20, 3, generator=generator)
    sampled = torch.randn(2, 20, 3, generator=generator)

    out = overlay_score_coordinates(sampled, target, torch.ones(20, dtype=torch.bool))

    assert out.shape == (2, 20, 3)
    torch.testing.assert_close(out, target.expand(2, -1, -1))


def test_overlay_places_missing_atoms_in_given_frame():
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(30, 3, generator=generator, dtype=torch.float64) * 5
    rotation = _random_rotation(generator)
    # The sample is the true structure in another frame.
    sampled = (
        target @ rotation + torch.tensor([10.0, -3.0, 7.0], dtype=torch.float64)
    )[None]
    mask = torch.ones(30, dtype=torch.bool)
    mask[[3, 17, 29]] = False

    out = overlay_score_coordinates(sampled.float(), target.float(), mask)

    torch.testing.assert_close(out[0], target.float(), atol=1e-4, rtol=0)


def test_validation_accepts_and_rejects_score_structure():
    job = _job("/x/dimer.cif", "ACDE", "GHIK", chain_ids=["A", "B"])
    validate_inference_jobs([job])

    bad_key = dict(job, scoreStructure={"structurePath": "/x.cif", "chain": "A"})
    with pytest.raises(ValueError, match="unknown keys"):
        validate_inference_jobs([bad_key])

    wrong_count = _job("/x/dimer.cif", "ACDE", "GHIK", chain_ids=["A"])
    with pytest.raises(ValueError, match="1 entries for 2 model chain"):
        validate_inference_jobs([wrong_count])

    ligand = _job("/x/dimer.cif", "ACDE")
    ligand["sequences"].append({"ligand": {"ligand": "CCD_ATP", "count": 1}})
    with pytest.raises(ValueError, match="only proteinChain"):
        validate_inference_jobs([ligand])
