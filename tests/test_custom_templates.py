# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from opendde.data.constants import ATOM37_ORDER
from opendde.data.inference.input_validation import (
    validate_custom_templates,
    validate_inference_jobs,
)
from opendde.data.template import custom_template
from opendde.data.template.custom_template import (
    load_custom_template_chain,
    make_custom_template_features,
    map_query_to_template,
)
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from runner.template_search import update_template_info

THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
}
# Rough backbone geometry; only non-collinear N/CA/C and a CB are needed.
ATOM_OFFSETS = {
    "N": (-1.2, 0.8, 0.0),
    "CA": (0.0, 0.0, 0.0),
    "C": (1.2, 0.8, 0.0),
    "O": (1.2, 2.0, 0.0),
    "CB": (0.0, -1.0, 1.2),
}


def _backbone_atoms(sequence: str, chain_id: str = "A", start: int = 1):
    atoms = []
    for i, code in enumerate(sequence):
        res_name = THREE_LETTER[code]
        for atom_name, offset in ATOM_OFFSETS.items():
            if res_name == "GLY" and atom_name == "CB":
                continue
            coord = np.array(offset) + np.array([3.8 * i, 0.5 * (i % 2), 0.0])
            atoms.append((chain_id, start + i, res_name, atom_name, coord))
    return atoms


def _write_pdb(path, atoms) -> None:
    atom_array = AtomArray(len(atoms))
    for i, (chain_id, res_id, res_name, atom_name, coord) in enumerate(atoms):
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = res_id
        atom_array.res_name[i] = res_name
        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = atom_name[0]
        atom_array.coord[i] = coord
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    pdb_file.write(str(path))


def _write_minimal_cif(path, atoms) -> None:
    """Write an mmCIF with only the _atom_site columns design tools tend to emit."""
    lines = [
        "data_design",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    for i, (chain_id, res_id, res_name, atom_name, coord) in enumerate(atoms, 1):
        group = "HETATM" if res_name in ("HOH", "MSE") else "ATOM"
        x, y, z = coord
        lines.append(
            f"{group} {i} {atom_name} {res_name} {chain_id} {res_id} "
            f"{x:.3f} {y:.3f} {z:.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _make_input(
    sequence: str, count: int, templates: list[dict]
) -> tuple[list[dict], AtomArray]:
    bioassembly = [
        {
            "proteinChain": {
                "sequence": sequence,
                "count": count,
                "templates": templates,
            }
        }
    ]
    n_res = len(sequence)
    atom_array = AtomArray(n_res * count)
    atom_array.res_id[:] = np.tile(np.arange(1, n_res + 1), count)
    atom_array.chain_id[:] = np.repeat([chr(ord("A") + i) for i in range(count)], n_res)
    atom_array.set_annotation(
        "asym_id_int", np.repeat(np.arange(count, dtype=np.int64), n_res)
    )
    atom_array.set_annotation("centre_atom_mask", np.ones(n_res * count, dtype=np.int8))
    return bioassembly, atom_array


def test_load_minimal_cif_without_entity_metadata(tmp_path):
    atoms = _backbone_atoms("ACDG")
    # Selenomethionine, a water and a ligand should be handled or skipped.
    atoms += [
        ("A", 5, "MSE", "N", np.array([20.0, 0.0, 0.0])),
        ("A", 5, "MSE", "CA", np.array([21.0, 0.0, 0.0])),
        ("A", 5, "MSE", "SE", np.array([22.0, 1.0, 0.0])),
        ("A", 101, "HOH", "O", np.array([0.0, 9.0, 9.0])),
        ("A", 102, "ATP", "PG", np.array([9.0, 9.0, 9.0])),
    ]
    path = tmp_path / "design.cif"
    _write_minimal_cif(path, atoms)

    chain = load_custom_template_chain(str(path))

    assert chain.sequence == "ACDGM"
    assert chain.atom_positions.shape == (5, 37, 3)
    assert chain.atom_mask[0, ATOM37_ORDER["CB"]] == 1
    assert chain.atom_mask[3, ATOM37_ORDER["CB"]] == 0  # glycine
    assert chain.atom_mask[4, ATOM37_ORDER["SD"]] == 1  # MSE selenium
    np.testing.assert_allclose(chain.atom_positions[4, ATOM37_ORDER["SD"]], [22, 1, 0])


def test_load_pdb_selects_requested_chain(tmp_path):
    path = tmp_path / "complex.pdb"
    _write_pdb(path, _backbone_atoms("ACDEF", "A") + _backbone_atoms("GHIK", "B"))

    assert load_custom_template_chain(str(path), "B").sequence == "GHIK"
    with pytest.raises(ValueError, match="set chainId"):
        load_custom_template_chain(str(path))
    with pytest.raises(ValueError, match="Chain 'Z' not found"):
        load_custom_template_chain(str(path), "Z")
    with pytest.raises(FileNotFoundError):
        load_custom_template_chain(str(tmp_path / "missing.pdb"))
    with pytest.raises(ValueError, match="Unsupported template structure format"):
        bad = tmp_path / "complex.xyz"
        bad.write_text("")
        load_custom_template_chain(str(bad))


def test_load_pdb_with_truncated_records_and_models(tmp_path):
    path = tmp_path / "opm.pdb"
    _write_pdb(path, _backbone_atoms("ACDEF", "R"))
    records = [
        line for line in path.read_text().splitlines() if line.startswith("ATOM")
    ]
    # OPM membrane pseudo-atoms end after the z coordinate.
    dummy = "HETATM 8156  N   DUM     2     -46.000 -10.000 -15.700"
    second_model = [line[:22] + " 999" + line[26:] for line in records]
    path.write_text(
        "\n".join(
            ["REMARK      1/2 of bilayer thickness:   15.7", "MODEL        1"]
            + records
            + [dummy, "ENDMDL", "MODEL        2"]
            + second_model
            + ["ENDMDL", "END"]
        )
        + "\n"
    )

    chain = load_custom_template_chain(str(path), "R")

    assert chain.sequence == "ACDEF"
    assert chain.atom_mask[:, ATOM37_ORDER["CA"]].sum() == 5


def test_map_query_to_template_without_kalign(monkeypatch):
    def fail_align(*args, **kwargs):
        raise AssertionError("Kalign should not be needed")

    monkeypatch.setattr(custom_template.Kalign, "__init__", fail_align)

    assert map_query_to_template("ACDEF", "ACDEF") == {i: i for i in range(5)}
    # Truncated template.
    assert map_query_to_template("ACDEFGHIK", "DEFG") == {2: 0, 3: 1, 4: 2, 5: 3}
    # Template longer than the query.
    assert map_query_to_template("DEF", "ACDEFG") == {0: 2, 1: 3, 2: 4}
    # Explicit indices win over sequence matching.
    assert map_query_to_template("ACDEF", "ACDEF", [0, 4], [1, 2]) == {0: 1, 4: 2}
    with pytest.raises(ValueError, match="templateIndices"):
        map_query_to_template("ACDEF", "ACD", [0], [3])


def test_map_query_to_template_falls_back_to_kalign(monkeypatch):
    calls = []

    class FakeKalign:
        def __init__(self, *, binary_path):
            calls.append(binary_path)

        def align(self, sequences):
            return ["ACDEF-", "AC-EFG"]

    monkeypatch.setattr(custom_template, "Kalign", FakeKalign)

    mapping = map_query_to_template("ACDEF", "ACEFG", kalign_binary_path="kalign")

    assert mapping == {0: 0, 1: 1, 3: 2, 4: 3}
    assert calls == ["kalign"]


def test_make_custom_template_features_truncation(tmp_path):
    path = tmp_path / "trunc.pdb"
    _write_pdb(path, _backbone_atoms("DEFG", start=40))

    feats = make_custom_template_features("ACDEFGHIK", {"structurePath": str(path)})

    assert feats["template_sequence"] == b"--DEFG---"
    ca_mask = feats["template_all_atom_masks"][:, ATOM37_ORDER["CA"]]
    np.testing.assert_array_equal(ca_mask, [0, 0, 1, 1, 1, 1, 0, 0, 0])
    present = feats["template_all_atom_masks"].astype(bool)
    np.testing.assert_allclose(
        feats["template_all_atom_positions"][present].mean(axis=0), 0, atol=1e-4
    )


def test_template_feature_uses_custom_templates_for_all_copies(tmp_path):
    sequence = "ACDEFGHIK"
    path = tmp_path / "design.cif"
    _write_minimal_cif(path, _backbone_atoms("CDEFGH"))
    templates = [{"structurePath": str(path), "chainId": "A"}]
    bioassembly, atom_array = _make_input(sequence, count=2, templates=templates)

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=bioassembly,
        atom_array=atom_array,
        use_template=True,
        online_template_featurizer=None,
    )

    n_token = 2 * len(sequence)
    assert features["template_aatype"].shape == (4, n_token)
    assert features["template_distogram"].shape == (4, n_token, n_token, 39)
    covered = np.zeros(len(sequence), dtype=bool)
    covered[1:7] = True
    for copy in range(2):
        tokens = slice(copy * len(sequence), (copy + 1) * len(sequence))
        pb_mask = features["template_pseudo_beta_mask"][0, tokens, tokens]
        np.testing.assert_array_equal(pb_mask, np.outer(covered, covered))
        assert features["template_distogram"][0, tokens, tokens].sum() > 0
    # Only one template was supplied; the remaining slots stay empty.
    assert features["template_pseudo_beta_mask"][1:].sum() == 0


def test_template_feature_ignores_custom_templates_when_disabled(tmp_path):
    path = tmp_path / "design.pdb"
    _write_pdb(path, _backbone_atoms("ACDEF"))
    bioassembly, atom_array = _make_input(
        "ACDEF", count=1, templates=[{"structurePath": str(path)}]
    )

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=bioassembly,
        atom_array=atom_array,
        use_template=False,
        online_template_featurizer=None,
    )

    assert features["template_pseudo_beta_mask"].sum() == 0


@pytest.mark.parametrize(
    ("templates", "message"),
    [
        ({"structurePath": "a.cif"}, "must be a list"),
        ([{"structurePath": "a.cif"}] * 5, "at most 4"),
        (["a.cif"], "must be an object"),
        ([{"mmcifPath": "a.cif"}], "unsupported field"),
        ([{"structurePath": ""}], "structurePath"),
        ([{"structurePath": "a.cif", "chainId": 1}], "chainId"),
        ([{"structurePath": "a.cif", "queryIndices": [0]}], "both queryIndices"),
        (
            [{"structurePath": "a.cif", "queryIndices": [0], "templateIndices": []}],
            "non-empty list",
        ),
        (
            [
                {
                    "structurePath": "a.cif",
                    "queryIndices": [0, True],
                    "templateIndices": [0, 1],
                }
            ],
            "non-negative integers",
        ),
        (
            [
                {
                    "structurePath": "a.cif",
                    "queryIndices": [0, 0],
                    "templateIndices": [0, 1],
                }
            ],
            "must not repeat",
        ),
        (
            [
                {
                    "structurePath": "a.cif",
                    "queryIndices": [0, 1],
                    "templateIndices": [0],
                }
            ],
            "same length",
        ),
    ],
)
def test_validate_custom_templates_rejects_invalid_specs(templates, message):
    with pytest.raises(ValueError, match=message):
        validate_custom_templates(templates, location="templates")


def test_validate_inference_jobs_checks_custom_templates():
    valid = {
        "structurePath": "design.cif",
        "chainId": "A",
        "queryIndices": [0, 1],
        "templateIndices": [3, 4],
    }
    job = {
        "name": "job",
        "sequences": [
            {"proteinChain": {"sequence": "ACD", "count": 1, "templates": [valid]}}
        ],
    }
    assert validate_inference_jobs([job]) == [job]

    job["sequences"] = [
        {"rnaSequence": {"sequence": "ACGU", "count": 1, "templates": [valid]}}
    ]
    with pytest.raises(ValueError, match="only supported for proteinChain"):
        validate_inference_jobs([job])


def test_update_template_info_skips_chains_with_custom_templates():
    json_data = [
        {
            "name": "job",
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "ACDEF",
                        "count": 1,
                        "templates": [{"structurePath": "design.cif"}],
                    }
                }
            ],
        }
    ]

    # No MSA is available, so a search would raise; custom templates skip it.
    assert update_template_info(json_data) is False
    assert "templatesPath" not in json_data[0]["sequences"][0]["proteinChain"]
