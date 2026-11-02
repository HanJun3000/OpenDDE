# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Score a user-supplied structure with the confidence head.

A job with ``scoreStructure`` keeps the normal trunk (sequence, MSA,
templates), but the confidence head sees the coordinates from the given file
instead of a diffusion sample. Atoms that the file lacks are filled from a
diffusion sample superposed onto the given structure (see
``overlay_score_coordinates``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from biotite.structure import AtomArray

from opendde.data.constants import ATOM37_ORDER
from opendde.data.template.custom_template import (
    list_protein_chains,
    load_custom_template_chain,
    map_query_to_template,
)
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

SCORE_COORDINATE_KEY = "score_coordinate"
SCORE_COORDINATE_MASK_KEY = "score_coordinate_mask"
# Atoms copied for mapped residues whose residue type differs from the query.
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def _model_protein_chains(sequences: Sequence[Mapping[str, Any]]) -> list[str]:
    """One query sequence per model chain, in model chain order."""
    chain_seqs = []
    for index, entity in enumerate(sequences):
        entity_type, body = next(iter(entity.items()))
        if entity_type != "proteinChain":
            raise ValueError(
                f"scoreStructure supports protein chains only; sequences[{index}] "
                f"is {entity_type}."
            )
        chain_seqs.extend([body["sequence"]] * int(body["count"]))
    return chain_seqs


def build_score_coordinates(
    job: Mapping[str, Any],
    atom_array: AtomArray,
    kalign_binary_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map the coordinates of ``job["scoreStructure"]`` onto the model atoms.

    Model chains (entities in JSON order, copies in order) are paired with the
    file chains listed in ``chainIds``, or with the file's protein chains in
    file order. Residues are mapped like custom templates (identity, unique
    substring, or Kalign). Mapped residues of the same type contribute all
    their atoms; mapped residues of a different type contribute backbone
    atoms only.

    Returns:
        Coordinates [N_atom, 3] (float32) and a mask [N_atom] (bool) of the
        model atoms that were found in the file.
    """
    spec = job["scoreStructure"]
    path = spec["structurePath"]
    chain_seqs = _model_protein_chains(job["sequences"])
    file_chains = spec.get("chainIds") or list_protein_chains(path)
    if len(file_chains) != len(chain_seqs):
        raise ValueError(
            f"scoreStructure {path}: {len(chain_seqs)} model chain(s) but "
            f"{len(file_chains)} file chain(s) {list(file_chains)}; set chainIds "
            "to one file chain per model chain."
        )

    coords = np.zeros((len(atom_array), 3), dtype=np.float32)
    mask = np.zeros(len(atom_array), dtype=bool)
    for asym_id, (query_seq, file_chain_id) in enumerate(zip(chain_seqs, file_chains)):
        chain = load_custom_template_chain(path, file_chain_id)
        mapping = map_query_to_template(
            query_seq, chain.sequence, kalign_binary_path=kalign_binary_path
        )
        n_identical = 0
        for q_idx, t_idx in mapping.items():
            identical = query_seq[q_idx] == chain.sequence[t_idx]
            n_identical += identical
            atom_idx = np.flatnonzero(
                (atom_array.asym_id_int == asym_id) & (atom_array.res_id == q_idx + 1)
            )
            for i in atom_idx:
                name = str(atom_array.atom_name[i])
                if name not in ATOM37_ORDER or not (
                    identical or name in _BACKBONE_ATOMS
                ):
                    continue
                slot = ATOM37_ORDER[name]
                if chain.atom_mask[t_idx, slot]:
                    coords[i] = chain.atom_positions[t_idx, slot]
                    mask[i] = True
        chain_atoms = atom_array.asym_id_int == asym_id
        logger.info(
            f"scoreStructure {path} chain {file_chain_id} -> model chain "
            f"{atom_array.chain_id[chain_atoms][0]}: {len(mapping)}/{len(query_seq)} "
            f"residues mapped ({n_identical} identical), "
            f"{int(mask[chain_atoms].sum())}/{int(chain_atoms.sum())} atoms."
        )

    if not mask.any():
        raise ValueError(f"scoreStructure {path}: no model atoms found in the file.")
    if not mask.all():
        logger.info(
            f"scoreStructure {path}: {int((~mask).sum())} model atom(s) missing in "
            "the file are filled from a superposed diffusion sample."
        )
    return coords, mask


def _kabsch_transform(
    mobile: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation and translation that superpose ``mobile`` [N, 3] onto ``target``."""
    mobile_center = mobile.mean(dim=0)
    target_center = target.mean(dim=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vh = torch.linalg.svd(covariance)
    sign = torch.sign(torch.linalg.det(u @ vh))
    correction = torch.diag(torch.stack([sign.new_ones(()), sign.new_ones(()), sign]))
    rotation = u @ correction @ vh
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def overlay_score_coordinates(
    sampled: torch.Tensor,
    score_coordinate: torch.Tensor,
    score_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replace sampled coordinates by the scored structure where it has atoms.

    Each sample [N_sample, N_atom, 3] is first superposed onto the scored
    atoms, so filled-in atoms sit in the frame of the given structure.
    """
    mask = score_mask.bool()
    target = score_coordinate.to(torch.float64)
    output = []
    for sample in sampled:
        sample64 = sample.to(torch.float64)
        if mask.sum() >= 3:
            rotation, translation = _kabsch_transform(sample64[mask], target[mask])
            sample64 = sample64 @ rotation + translation
        sample64[mask] = target[mask]
        output.append(sample64.to(sampled.dtype))
    return torch.stack(output, dim=0)
