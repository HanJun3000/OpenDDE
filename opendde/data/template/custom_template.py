# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""User-supplied structure templates.

Custom templates are protein chains read directly from local PDB or mmCIF
files (e.g. truncations, edited structures, de novo designs, or predicted
models). Unlike search hits, they need no PDB ID, release date, or SEQRES
records: the template sequence is taken from the residues that are present in
the file.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import io
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import biotite.structure.info as struc_info
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np

from opendde.data.constants import ATOM37_NUM, ATOM37_ORDER, PROTEIN_CHAIN
from opendde.data.template.template_parser import encode_template_restype
from opendde.data.tools.kalign import Kalign
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

# Custom templates have no PDB release date; this placeholder only satisfies
# the shared template feature packaging and is not used by the model.
_CUSTOM_TEMPLATE_RELEASE_DATE = "1970-01-01"
_CIF_SUFFIXES = (".cif", ".mmcif")
_PDB_SUFFIXES = (".pdb", ".ent")
_MISSING_CIF_VALUES = (".", "?")


@dataclasses.dataclass(frozen=True)
class CustomTemplateChain:
    """Observed amino-acid residues of one chain in a template structure."""

    sequence: str
    # atom37 coordinates and mask: [N_res, 37, 3] and [N_res, 37]
    atom_positions: np.ndarray
    atom_mask: np.ndarray


def _read_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.read()
    with open(path, "r") as f:
        return f.read()


def _structure_format(path: str) -> str:
    name = path.lower()
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    if name.endswith(_CIF_SUFFIXES):
        return "cif"
    if name.endswith(_PDB_SUFFIXES):
        return "pdb"
    raise ValueError(
        f"Unsupported template structure format: {path}. Use .cif, .mmcif, "
        ".pdb, or .ent (optionally gzipped)."
    )


def _cif_column(
    atom_site: Any, names: Sequence[str], n_atom: int, default: str = ""
) -> np.ndarray:
    """Return the first present column, treating '.'/'?' as missing values."""
    for name in names:
        if name in atom_site:
            values = atom_site[name].as_array(str)
            if not np.all(np.isin(values, _MISSING_CIF_VALUES)):
                return values
    return np.full(n_atom, default, dtype=object)


def _read_cif_atoms(text: str) -> Dict[str, np.ndarray]:
    """Read ``_atom_site`` leniently; design tools often omit most columns."""
    block = pdbx.CIFFile.read(io.StringIO(text)).block
    if "atom_site" not in block:
        raise ValueError("mmCIF file has no _atom_site records.")
    atom_site = block["atom_site"]
    n_atom = atom_site.row_count
    atoms = {
        "chain_id": _cif_column(atom_site, ("auth_asym_id", "label_asym_id"), n_atom),
        "res_id": _cif_column(atom_site, ("auth_seq_id", "label_seq_id"), n_atom),
        "ins_code": _cif_column(atom_site, ("pdbx_PDB_ins_code",), n_atom),
        "res_name": _cif_column(atom_site, ("auth_comp_id", "label_comp_id"), n_atom),
        "atom_name": _cif_column(atom_site, ("auth_atom_id", "label_atom_id"), n_atom),
        "coord": np.stack(
            [atom_site[f"Cartn_{axis}"].as_array(np.float32) for axis in "xyz"],
            axis=-1,
        ),
    }
    model = _cif_column(atom_site, ("pdbx_PDB_model_num",), n_atom)
    keep = model == model[0]
    return {key: value[keep] for key, value in atoms.items()}


def _read_pdb_atoms(text: str) -> Dict[str, np.ndarray]:
    atom_array = pdb.PDBFile.read(io.StringIO(text)).get_structure(
        model=1, altloc="first"
    )
    return {
        "chain_id": atom_array.chain_id,
        "res_id": atom_array.res_id.astype(str),
        "ins_code": atom_array.ins_code,
        "res_name": atom_array.res_name,
        "atom_name": atom_array.atom_name,
        "coord": atom_array.coord.astype(np.float32),
    }


@functools.lru_cache(maxsize=1)
def _amino_acid_names() -> frozenset:
    return frozenset(struc_info.amino_acid_names())


@functools.lru_cache(maxsize=None)
def _one_letter_code(res_name: str) -> Optional[str]:
    """Return the one-letter code of an amino acid, or None for other residues."""
    if res_name not in _amino_acid_names():
        return None
    code = struc_info.one_letter_code(res_name)
    return code if code and len(code) == 1 else "X"


def load_custom_template_chain(
    structure_path: str, chain_id: Optional[str] = None
) -> CustomTemplateChain:
    """
    Read the observed amino-acid residues of one chain from a PDB/mmCIF file.

    Args:
        structure_path: Path to a .cif/.mmcif/.pdb/.ent file, optionally gzipped.
        chain_id: Author chain ID. May be omitted if the file has exactly one
            protein chain.

    Returns:
        The chain's one-letter sequence (in file order) with atom37 coordinates.
    """
    if not os.path.isfile(structure_path):
        raise FileNotFoundError(f"Template structure not found: {structure_path}")
    text = _read_text(structure_path)
    if _structure_format(structure_path) == "cif":
        atoms = _read_cif_atoms(text)
    else:
        atoms = _read_pdb_atoms(text)

    # Group atoms into residues in order of appearance, per chain.
    residues: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    for chain, res_id, ins_code, res_name, atom_name, coord in zip(
        atoms["chain_id"],
        atoms["res_id"],
        atoms["ins_code"],
        atoms["res_name"],
        atoms["atom_name"],
        atoms["coord"],
    ):
        chain, res_name = str(chain), str(res_name)
        code = _one_letter_code(res_name)
        if code is None:
            continue
        ins_code = "" if ins_code in _MISSING_CIF_VALUES else str(ins_code)
        residue = residues.setdefault(chain, {}).setdefault(
            (str(res_id), ins_code), {"code": code, "atoms": {}}
        )
        atom_name = str(atom_name)
        if atom_name == "SE" and res_name == "MSE":
            atom_name = "SD"
        # Keep the first occurrence, i.e. the first alternate location.
        if atom_name in ATOM37_ORDER and atom_name not in residue["atoms"]:
            residue["atoms"][atom_name] = coord

    if not residues:
        raise ValueError(f"No protein chains found in {structure_path}.")
    if chain_id is None:
        if len(residues) != 1:
            raise ValueError(
                f"{structure_path} has several protein chains {sorted(residues)}; "
                "set chainId to select one."
            )
        chain_id = next(iter(residues))
    if chain_id not in residues:
        raise ValueError(
            f"Chain {chain_id!r} not found among protein chains "
            f"{sorted(residues)} in {structure_path}."
        )

    chain_residues = list(residues[chain_id].values())
    n_res = len(chain_residues)
    atom_positions = np.zeros((n_res, ATOM37_NUM, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, ATOM37_NUM), dtype=np.float32)
    for i, residue in enumerate(chain_residues):
        for atom_name, coord in residue["atoms"].items():
            atom_positions[i, ATOM37_ORDER[atom_name]] = coord
            atom_mask[i, ATOM37_ORDER[atom_name]] = 1.0
    sequence = "".join(residue["code"] for residue in chain_residues)
    return CustomTemplateChain(sequence, atom_positions, atom_mask)


def map_query_to_template(
    query_seq: str,
    template_seq: str,
    query_indices: Optional[Sequence[int]] = None,
    template_indices: Optional[Sequence[int]] = None,
    kalign_binary_path: Optional[str] = None,
) -> Dict[int, int]:
    """
    Map 0-based query residue indices to 0-based template residue indices.

    Explicit indices take precedence. Otherwise an exact or unique substring
    match (e.g. a truncated template) is used, and Kalign aligns the two
    sequences only when neither applies.
    """
    if query_indices is not None and template_indices is not None:
        for name, indices, length in (
            ("queryIndices", query_indices, len(query_seq)),
            ("templateIndices", template_indices, len(template_seq)),
        ):
            out_of_range = [i for i in indices if i >= length]
            if out_of_range:
                raise ValueError(
                    f"{name} {out_of_range[:5]} out of range for length {length}."
                )
        return dict(zip(query_indices, template_indices))

    if query_seq.count(template_seq) == 1:
        offset = query_seq.index(template_seq)
        return {offset + i: i for i in range(len(template_seq))}
    if template_seq.count(query_seq) == 1:
        offset = template_seq.index(query_seq)
        return {i: offset + i for i in range(len(query_seq))}

    q_aln, t_aln = Kalign(binary_path=kalign_binary_path).align(
        [query_seq, template_seq]
    )
    mapping = {}
    q_idx, t_idx = -1, -1
    for qa, ta in zip(q_aln, t_aln):
        if qa != "-":
            q_idx += 1
        if ta != "-":
            t_idx += 1
        if qa != "-" and ta != "-":
            mapping[q_idx] = t_idx
    return mapping


def make_custom_template_features(
    query_seq: str,
    spec: Mapping[str, Any],
    kalign_binary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build per-hit template features for one custom template.

    The returned dict has the same fields as a processed search hit, so it can
    be packaged with ``TemplateFeatures.package_template_features``.
    """
    structure_path = spec["structurePath"]
    chain_id = spec.get("chainId")
    chain = load_custom_template_chain(structure_path, chain_id)
    mapping = map_query_to_template(
        query_seq,
        chain.sequence,
        query_indices=spec.get("queryIndices"),
        template_indices=spec.get("templateIndices"),
        kalign_binary_path=kalign_binary_path,
    )

    num_query = len(query_seq)
    out_pos = np.zeros((num_query, ATOM37_NUM, 3), dtype=np.float32)
    out_mask = np.zeros((num_query, ATOM37_NUM), dtype=np.float32)
    out_seq = ["-"] * num_query
    for q_idx, t_idx in mapping.items():
        out_pos[q_idx] = chain.atom_positions[t_idx]
        out_mask[q_idx] = chain.atom_mask[t_idx]
        out_seq[q_idx] = chain.sequence[t_idx]

    mask_bool = out_mask.astype(bool)
    if not mask_bool[:, ATOM37_ORDER["CA"]].any():
        raise ValueError(
            f"Custom template {structure_path} (chain {chain_id or 'auto'}) "
            "has no aligned residues with coordinates."
        )
    out_pos[mask_bool] -= out_pos[mask_bool].mean(axis=0)

    n_identical = sum(
        query_seq[q_idx] == chain.sequence[t_idx] for q_idx, t_idx in mapping.items()
    )
    logger.info(
        f"Custom template {structure_path} (chain {chain_id or 'auto'}): "
        f"{len(mapping)}/{num_query} query residues mapped, "
        f"{n_identical} identical."
    )

    out_seq_str = "".join(out_seq)
    name = f"custom:{os.path.basename(structure_path)}_{chain_id or ''}"
    return {
        "template_all_atom_positions": out_pos,
        "template_all_atom_masks": out_mask,
        "template_sequence": out_seq_str.encode(),
        "template_aatype": np.array(
            encode_template_restype(PROTEIN_CHAIN, out_seq_str), dtype=np.int32
        ),
        "template_domain_names": np.array(name.encode(), dtype=object),
        "template_release_date": np.array(
            _CUSTOM_TEMPLATE_RELEASE_DATE.encode(), dtype=object
        ),
    }
