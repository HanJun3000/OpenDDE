# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Validation shared by inference loading and output path construction."""

from __future__ import annotations

from typing import Any

_MAX_NUMPY_SEED = 2**32 - 1
MAX_CUSTOM_TEMPLATES = 4
CUSTOM_TEMPLATE_KEYS = frozenset(
    {"structurePath", "chainId", "queryIndices", "templateIndices"}
)


def validate_inference_seed(value: Any, *, location: str = "seed") -> int:
    """Validate a seed before it reaches NumPy/PyTorch RNG setup."""

    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer, not a boolean.")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.strip() == value and value.isdecimal():
        # Preserve compatibility with older JSON/CLI inputs that quoted seeds.
        seed = int(value)
    else:
        raise ValueError(f"{location} must be an integer; got {value!r}.")
    if not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"{location} must be in [0, {_MAX_NUMPY_SEED}]; got {seed}.")
    return seed


def validate_sample_name(value: Any, *, job_index: int | None = None) -> str:
    """Return a safe output path component for one inference job."""

    location = f" for job {job_index}" if job_index is not None else ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Inference job name{location} must be a non-empty string.")
    if value.casefold() == "err":
        raise ValueError(
            f"Inference job name{location} {value!r} is reserved for error reports."
        )
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(
            f"Inference job name{location} must be a single safe path component; "
            f"got {value!r}."
        )
    return value


def validate_inference_jobs(value: Any) -> list[dict[str, Any]]:
    """Validate the top-level inference job list and collision-sensitive fields."""

    if not isinstance(value, list) or not value:
        raise ValueError(
            "Input JSON must be a non-empty top-level list, "
            f"got {type(value).__name__}."
        )

    jobs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, job in enumerate(value):
        if not isinstance(job, dict):
            raise ValueError(
                f"Inference job {index} must be an object, got {type(job).__name__}."
            )
        name = validate_sample_name(job.get("name"), job_index=index)
        if name in seen_names:
            raise ValueError(
                f"Inference job name {name!r} is duplicated in the same input JSON; "
                "duplicate names would overwrite outputs."
            )
        seen_names.add(name)

        model_seeds = job.get("modelSeeds")
        if model_seeds is not None:
            if not isinstance(model_seeds, list):
                raise ValueError(
                    f"modelSeeds for job {name!r} must be a list of integers."
                )
            for seed in model_seeds:
                try:
                    validate_inference_seed(
                        seed,
                        location=f"modelSeeds for job {name!r}",
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"modelSeeds for job {name!r} contains invalid seed {seed!r}."
                    ) from exc
        _validate_custom_templates(job, name)
        jobs.append(job)
    return jobs


def _validate_custom_templates(job: dict[str, Any], name: str) -> None:
    """Validate user-supplied structure templates before any heavy work runs."""

    sequences = job.get("sequences")
    if not isinstance(sequences, list):
        return
    for index, entity in enumerate(sequences):
        if not isinstance(entity, dict):
            continue
        for entity_type, body in entity.items():
            if not isinstance(body, dict) or "templates" not in body:
                continue
            location = f"sequences[{index}].{entity_type}.templates for job {name!r}"
            if entity_type != "proteinChain":
                raise ValueError(
                    f"{location}: templates are only supported for proteinChain."
                )
            validate_custom_templates(body["templates"], location=location)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_custom_templates(value: Any, *, location: str) -> None:
    """Validate the JSON shape of a ``proteinChain.templates`` entry."""

    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list of template objects.")
    if len(value) > MAX_CUSTOM_TEMPLATES:
        raise ValueError(
            f"{location} has {len(value)} templates; at most "
            f"{MAX_CUSTOM_TEMPLATES} are supported."
        )
    for index, spec in enumerate(value):
        spec_location = f"{location}[{index}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{spec_location} must be an object.")
        unknown = sorted(set(spec) - CUSTOM_TEMPLATE_KEYS)
        if unknown:
            raise ValueError(
                f"{spec_location} has unsupported field(s) {unknown}; allowed "
                f"fields are {sorted(CUSTOM_TEMPLATE_KEYS)}."
            )
        path = spec.get("structurePath")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{spec_location}.structurePath must be a non-empty path.")
        chain_id = spec.get("chainId")
        if chain_id is not None and (not isinstance(chain_id, str) or not chain_id):
            raise ValueError(f"{spec_location}.chainId must be a non-empty string.")

        query_indices = spec.get("queryIndices")
        template_indices = spec.get("templateIndices")
        if (query_indices is None) != (template_indices is None):
            raise ValueError(
                f"{spec_location} must give both queryIndices and templateIndices, "
                "or neither."
            )
        if query_indices is None:
            continue
        for key, indices in (
            ("queryIndices", query_indices),
            ("templateIndices", template_indices),
        ):
            if not isinstance(indices, list) or not indices:
                raise ValueError(f"{spec_location}.{key} must be a non-empty list.")
            if not all(_is_non_negative_int(i) for i in indices):
                raise ValueError(
                    f"{spec_location}.{key} must contain non-negative integers."
                )
            if len(set(indices)) != len(indices):
                raise ValueError(f"{spec_location}.{key} must not repeat indices.")
        if len(query_indices) != len(template_indices):
            raise ValueError(
                f"{spec_location}.queryIndices and templateIndices must have the "
                "same length."
            )
