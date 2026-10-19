#!/usr/bin/env python3
"""
Single-problem difficulty analysis interface for OptMATH / Opt-Arena.

Pipeline:
1. Generate mathematical modeling for an input question when modeling is absent.
2. Split the question/modeling pair into atomic information and related constraints.
3. Classify the problem type.
4. Classify each related constraint, then attach generic atom_info by
   (problem_type, constraint_types).
5. Calculate constraint, atomic-information, and overall problem difficulty.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime with a clear error
    OpenAI = None


OPTMATH_DIR = Path(__file__).resolve().parent
DEFAULT_EVO_OPT_ARENA_DIR = Path(
    os.getenv("EVO_OPT_ARENA_DIR", str(Path(__file__).resolve().parent))
)
DEFAULT_PROBLEM_TYPES_FILE = DEFAULT_EVO_OPT_ARENA_DIR / "problem_type.json"
DEFAULT_CONSTRAINT_CATALOG_FILE = (
    DEFAULT_EVO_OPT_ARENA_DIR / "constraints_with_difficulty_scores-train.json"
)
FALLBACK_CONSTRAINT_CATALOG_FILE = (
    DEFAULT_EVO_OPT_ARENA_DIR / "constraints_with_difficulty.json"
)
DEFAULT_ATOMIC_GROUPS_FILE = DEFAULT_EVO_OPT_ARENA_DIR / "atomic_info_all-2.json"
DEFAULT_KERNEL_COMBINATIONS_FILE = DEFAULT_EVO_OPT_ARENA_DIR / "kernel_combinations.json"

DEFAULT_MODEL = os.getenv("OPT_ARENA_MODEL") or os.getenv("DEEPSEEK_MODEL") or "gpt-4o-mini"
DEFAULT_BASE_URL = (
    os.getenv("OPT_ARENA_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
)


MODELING_PROMPT_TEMPLATE = """READ a natural language optimization problem description and FORMULATE a clear and correct mathematical optimization model.

STRICT REQUIREMENTS:
1. Output EXACTLY three sections:
   - "### Decision Variables"
   - "### Objective Function"
   - "### Constraints"
2. Extract decision variables, objective, and constraints from the problem description.
3. Keep formulas readable with simple inline mathematical notation.
4. The model should be linear or mixed-integer linear whenever possible.
5. Do not include explanations outside the three requested sections.

Input:
{question}

Output:
"""


ATOMIC_INFO_PROMPT_TEMPLATE = """You are a professional operations research and mathematical modeling expert.

Your task is to extract fine-grained atomic information from a given optimization problem and match each item with constraints in a provided mathematical model.

Requirements:
1. Split the question into minimal, indivisible atomic information units.
2. Only extract information related to constraints, not the objective function.
3. For each atomic information item, identify the related mathematical constraint(s) from the modeling.
4. Return ONLY valid JSON in this format:
[
  {{
    "info": "<atomic information>",
    "related_constraints": ["<constraint expression>"]
  }}
]

Question:
{question}

Modeling:
{modeling}
"""


CONSTRAINT_CLASSIFICATION_PROMPT_TEMPLATE = """You are an expert in operations research and mathematical optimization.

Classify each constraint into exactly one of the available constraint types. Also determine whether the constraint is one-dimensional or multi-dimensional.

Problem type:
{problem_type}

Available constraint types:
{constraint_catalog}

Constraints to classify:
{constraints}

Return ONLY valid JSON in this format:
[
  {{
    "constraint": "<same constraint expression>",
    "type": "<one available type or Unclassified>",
    "dimensionality": "one-dimensional"
  }}
]

Dimensionality guide:
- one-dimensional: variables have no subscripts or a single index, e.g. x_i, y_j.
- multi-dimensional: variables use multiple indices, e.g. x_ij, x_{{i,j}}, y_{{i,j,t}}.
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install it before using the LLM pipeline.")

    resolved_key = (
        api_key
        or os.getenv("OPT_ARENA_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError(
            "Missing API key. Set OPT_ARENA_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY."
        )

    kwargs = {"api_key": resolved_key}
    resolved_base_url = base_url or DEFAULT_BASE_URL
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def call_llm_json(
    client: Any,
    model: str,
    prompt: str,
    *,
    expected_type: type,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> Any:
    text = call_llm_text(
        client,
        model,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_json_response(text, expected_type=expected_type)


def call_llm_text(
    client: Any,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an operations research expert. Follow output format instructions exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(text: str, *, expected_type: type) -> Any:
    cleaned_text = strip_markdown_code_fence(text)
    candidates = [cleaned_text, text]
    if expected_type is list:
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            candidates.append(match.group(0))
    elif expected_type is dict:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            candidates.append(match.group(0))

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, expected_type):
                return parsed
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to parse JSON response as {expected_type.__name__}: {text}") from last_error


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S | re.I)
    return fence_match.group(1).strip() if fence_match else stripped


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def normalize_key(text: Any) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[\t\n\r]", " ", text)
    text = re.sub(r"[\.,;:!\?\(\)\[\]\{\}\"']", "", text)
    return " ".join(text.split())


def generate_modeling(question: str, client: Any, model: str = DEFAULT_MODEL) -> str:
    prompt = MODELING_PROMPT_TEMPLATE.format(question=question)
    return call_llm_text(client, model, prompt, temperature=0.2, max_tokens=4000)


def extract_atomic_information(
    question: str,
    modeling: str,
    client: Any,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    prompt = ATOMIC_INFO_PROMPT_TEMPLATE.format(question=question, modeling=modeling)
    raw_items = call_llm_json(
        client,
        model,
        prompt,
        expected_type=list,
        temperature=0.1,
        max_tokens=3000,
    )

    atomic_information = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        info = normalize_text(item.get("info", ""))
        related_constraints = [
            normalize_text(c) for c in item.get("related_constraints", []) if normalize_text(c)
        ]
        if info:
            atomic_information.append(
                {"info": info, "related_constraints": related_constraints}
            )
    return atomic_information


def build_problem_classification_prompt(
    question: str,
    problem_types: Sequence[Dict[str, Any]],
) -> str:
    type_descriptions = []
    for problem_type in problem_types:
        type_descriptions.append(
            "Problem Type: {problem}\nDescription: {description}\nExample: {example}".format(
                problem=problem_type.get("problem", ""),
                description=problem_type.get("description", ""),
                example=problem_type.get("example", ""),
            )
        )
    return f"""You are an expert in optimization problem classification.

Classify the following optimization problem into one of the provided categories.

Problem:
{question}

Available problem types:
{chr(10).join(type_descriptions)}

You MUST choose exactly one type from the Available problem types.
If none is a perfect fit, choose the closest or most similar type.
Respond ONLY with the problem type name exactly as it appears in the "Problem Type" field.
"""


def classify_problem_type(
    question: str,
    problem_types: Sequence[Dict[str, Any]],
    client: Any,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_problem_classification_prompt(question, problem_types)
    result = call_llm_text(client, model, prompt, temperature=0.1, max_tokens=80)
    allowed_types = [item.get("problem") for item in problem_types if item.get("problem")]
    result = result.strip().strip('"').strip("'")
    return match_problem_type(result, allowed_types, question=question)


def match_problem_type(
    raw_result: str,
    allowed_types: Sequence[str],
    question: str = "",
) -> str:
    if not allowed_types:
        return "Unknown"

    cleaned = normalize_text(raw_result).strip().strip('"').strip("'")
    if cleaned in allowed_types:
        return cleaned

    heuristic_match = infer_problem_type_from_question(question, allowed_types)
    if normalize_key(cleaned) in {"unknown", "none", "n/a", "not sure", ""}:
        return heuristic_match or allowed_types[0]

    normalized_to_type = {normalize_key(problem_type): problem_type for problem_type in allowed_types}
    normalized_cleaned = normalize_key(cleaned)
    if normalized_cleaned in normalized_to_type:
        return normalized_to_type[normalized_cleaned]

    for problem_type in allowed_types:
        if problem_type in cleaned:
            return problem_type

    if heuristic_match:
        return heuristic_match

    matches = difflib.get_close_matches(
        normalized_cleaned,
        list(normalized_to_type.keys()),
        n=1,
        cutoff=0.0,
    )
    if matches:
        return normalized_to_type[matches[0]]

    return allowed_types[0]


def infer_problem_type_from_question(question: str, allowed_types: Sequence[str]) -> Optional[str]:
    text = normalize_key(question)
    allowed = set(allowed_types)

    candidates = [
        (
            "Assignment Problem",
            ["assign each", "assigned to exactly one", "each task", "each worker"],
        ),
        (
            "Knapsack Problem",
            ["backpack", "knapsack", "select items", "selected weight", "total value"],
        ),
        (
            "Production Planning Problem",
            ["produce", "production", "make two", "make at least", "pills", "factory"],
        ),
        (
            "Resource Allocation Linear Programming",
            ["limited resource", "allocate", "resource", "transport", "boat", "canoe", "minimize total amount of time"],
        ),
        (
            "Diet Problem",
            ["diet", "nutrient", "calories", "protein"],
        ),
        (
            "Blending Problem",
            ["blend", "mixture", "ingredient mix"],
        ),
    ]

    for problem_type, keywords in candidates:
        if problem_type in allowed and any(keyword in text for keyword in keywords):
            return problem_type
    return None


def format_constraint_catalog(constraint_catalog: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for item in constraint_catalog:
        lines.append(
            "- Type: {type}\n  Description: {description}\n  Example: {example}".format(
                type=item.get("type", ""),
                description=item.get("description", ""),
                example=item.get("example", ""),
            )
        )
    return "\n".join(lines)


def unique_constraints(atomic_information: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    constraints = []
    for atomic_info in atomic_information:
        for constraint in atomic_info.get("related_constraints", []):
            key = normalize_text(constraint)
            if key and key not in seen:
                seen.add(key)
                constraints.append(key)
    return constraints


def unique_constraint_sources(
    atomic_information: Sequence[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    constraints = []
    seen = set()
    sources: Dict[str, List[str]] = {}

    for atomic_info in atomic_information:
        source_info = normalize_text(atomic_info.get("info", ""))
        for constraint in atomic_info.get("related_constraints", []):
            constraint_text = normalize_text(constraint)
            if not constraint_text:
                continue
            if constraint_text not in seen:
                seen.add(constraint_text)
                constraints.append(constraint_text)
            if source_info:
                sources.setdefault(constraint_text, [])
                if source_info not in sources[constraint_text]:
                    sources[constraint_text].append(source_info)

    return constraints, sources


def classify_constraints(
    constraints: Sequence[str],
    problem_type: str,
    constraint_catalog: Sequence[Dict[str, Any]],
    client: Any,
    model: str = DEFAULT_MODEL,
    constraint_sources: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not constraints:
        return {}

    allowed_types = {item.get("type") for item in constraint_catalog}
    constraint_lines = []
    for idx, constraint in enumerate(constraints):
        source_infos = []
        if constraint_sources:
            source_infos = constraint_sources.get(constraint, [])
        source_text = "; ".join(source_infos) if source_infos else "N/A"
        constraint_lines.append(
            f"{idx + 1}. Constraint Expression: {constraint}\n"
            f"   Source Information: {source_text}"
        )

    prompt = CONSTRAINT_CLASSIFICATION_PROMPT_TEMPLATE.format(
        problem_type=problem_type,
        constraint_catalog=format_constraint_catalog(constraint_catalog),
        constraints="\n".join(constraint_lines),
    )
    raw_items = call_llm_json(
        client,
        model,
        prompt,
        expected_type=list,
        temperature=0.1,
        max_tokens=4000,
    )

    classified: Dict[str, Dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        constraint = normalize_text(item.get("constraint", ""))
        constraint_type = normalize_text(item.get("type", "Unclassified"))
        dimensionality = normalize_text(item.get("dimensionality", "unknown"))

        if constraint not in constraints:
            matched_constraint = match_constraint_text(constraint, constraints)
            if not matched_constraint:
                continue
            constraint = matched_constraint
        if constraint_type not in allowed_types:
            constraint_type = "Unclassified"
        if dimensionality not in {"one-dimensional", "multi-dimensional"}:
            dimensionality = infer_dimensionality(constraint)

        source_infos = constraint_sources.get(constraint, []) if constraint_sources else []
        inferred_type = infer_constraint_type(
            constraint,
            source_infos,
            allowed_types,
        )
        if should_use_inferred_constraint_type(constraint_type, inferred_type):
            constraint_type = inferred_type

        classified[constraint] = {
            "constraint": constraint,
            "type": constraint_type,
            "dimensionality": dimensionality,
        }

    for constraint in constraints:
        source_infos = constraint_sources.get(constraint, []) if constraint_sources else []
        classified.setdefault(
            constraint,
            {
                "constraint": constraint,
                "type": infer_constraint_type(constraint, source_infos, allowed_types),
                "dimensionality": infer_dimensionality(constraint),
            },
        )
    return classified


def infer_constraint_type(
    constraint: str,
    source_infos: Sequence[str],
    allowed_types: Sequence[str],
) -> str:
    allowed = set(allowed_types)
    text = normalize_ascii_math(constraint)
    source_text = normalize_ascii_math(" ".join(source_infos))
    combined = f"{text} {source_text}"

    if "Binary domain constraint" in allowed:
        if re.search(r"\{0\s*,\s*1\}", text) or "binary" in combined:
            return "Binary domain constraint"

    if "Non-Negativity constraint" in allowed:
        if is_nonnegativity_constraint(text, source_text):
            return "Non-Negativity constraint"

    if "Ratio constraint" in allowed:
        if (
            "%" in combined
            or "percent" in combined
            or "ratio" in combined
            or "proportion" in combined
            or "at least 60" in combined
        ):
            return "Ratio constraint"

    if "Upper-bound constraint" in allowed:
        if is_simple_upper_bound_constraint(text, source_text):
            return "Upper-bound constraint"

    return "Unclassified"


def should_use_inferred_constraint_type(current_type: str, inferred_type: str) -> bool:
    if inferred_type == "Unclassified":
        return False
    if current_type == "Unclassified":
        return True
    if inferred_type == "Ratio constraint" and current_type in {
        "Linear difference constraint",
        "Pairwise comparison constraint",
    }:
        return True
    return False


def normalize_ascii_math(text: Any) -> str:
    normalized = normalize_text(text).lower()
    replacements = {
        "≤": "<=",
        "\\leq": "<=",
        "\\le": "<=",
        "≥": ">=",
        "\\geq": ">=",
        "\\ge": ">=",
        "\\in": "in",
        "∈": "in",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized


def is_nonnegativity_constraint(text: str, source_text: str) -> bool:
    if "non-negativity" in source_text or "nonnegative" in source_text or "non-negative" in source_text:
        return True

    compact = text.replace(" ", "")
    if re.fullmatch(r"[a-z](?:_[a-z0-9]+|\{[^}]+\})?>=0", compact):
        return True
    if re.fullmatch(r"(?:[a-z](?:_[a-z0-9]+|\{[^}]+\})?,)+[a-z](?:_[a-z0-9]+|\{[^}]+\})?>=0", compact):
        return True
    return False


def is_simple_upper_bound_constraint(text: str, source_text: str) -> bool:
    if "at most" in source_text or "cannot exceed" in source_text or "no more than" in source_text:
        return True

    compact = text.replace(" ", "")
    return bool(re.fullmatch(r"[a-z](?:_[a-z0-9]+|\{[^}]+\})?<=-?\d+(?:\.\d+)?", compact))


def match_constraint_text(candidate: str, constraints: Sequence[str]) -> Optional[str]:
    candidate_key = normalize_key(candidate)
    for constraint in constraints:
        if normalize_key(constraint) == candidate_key:
            return constraint
    return None


def infer_dimensionality(constraint: str) -> str:
    # x_{i,j}, x_ij, x[i,j], and x_{i,j,t} are treated as multi-dimensional.
    patterns = [
        r"[A-Za-z]\s*_\s*\{[^}]*,[^}]*\}",
        r"[A-Za-z]\s*\[[^\]]*,[^\]]*\]",
        r"[A-Za-z]\s*_[A-Za-z0-9]{2,}",
    ]
    if any(re.search(pattern, constraint) for pattern in patterns):
        return "multi-dimensional"
    return "one-dimensional"


def attach_constraint_types_to_atomic_information(
    atomic_information: Sequence[Dict[str, Any]],
    classified_constraints: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched_items = []
    for atomic_info in atomic_information:
        enriched = dict(atomic_info)
        constraints_with_types = []
        for constraint in atomic_info.get("related_constraints", []):
            constraint_key = match_constraint_text(constraint, classified_constraints.keys()) or constraint
            classification = classified_constraints.get(constraint_key)
            if classification:
                constraints_with_types.append(dict(classification))
        enriched["constraints_with_types"] = constraints_with_types
        enriched_items.append(enriched)
    return enriched_items


def load_atomic_group_map(path: Path = DEFAULT_ATOMIC_GROUPS_FILE) -> Dict[Tuple[str, str], Dict[str, Any]]:
    groups = load_json(path)
    group_map = {}
    for group in groups:
        problem_type = group.get("problem_type", "Unknown")
        constraint_key = constraint_types_key(group.get("constraint_types", []))
        group_map[(problem_type, constraint_key)] = group
    return group_map


def load_kernel_data(path: Path = DEFAULT_KERNEL_COMBINATIONS_FILE) -> Dict[str, Any]:
    if not path or not path.exists():
        return {"kernels": {}, "combinations_frequency": {}}
    return load_json(path)


def constraint_types_key(constraint_types: Iterable[str]) -> str:
    return "|".join(sorted(t for t in constraint_types if t))


def attach_atom_info(
    atomic_information: Sequence[Dict[str, Any]],
    problem_type: str,
    atomic_group_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched_items = []
    for atomic_info in atomic_information:
        enriched = dict(atomic_info)
        constraint_types = [
            item.get("type")
            for item in atomic_info.get("constraints_with_types", [])
            if item.get("type") and item.get("type") != "Unclassified"
        ]
        group = atomic_group_map.get((problem_type, constraint_types_key(constraint_types)))
        enriched["atom_info"] = group.get("merged_info") if group else "No matching generic description found"
        enriched_items.append(enriched)
    return enriched_items


def split_kernel_combo(combo_key: str) -> List[str]:
    parts = []
    chunks = combo_key.split(".,") if ".," in combo_key else [combo_key]
    for chunk in chunks:
        cleaned = chunk.strip().rstrip(".").strip()
        if cleaned:
            parts.append(f"{cleaned}.")
    return parts


def normalize_atom_info_text(text: Any) -> str:
    normalized = normalize_text(text).strip()
    return normalized.rstrip(".")


def get_problem_kernel(
    atomic_information: Sequence[Dict[str, Any]],
    problem_type: str,
    kernel_data: Dict[str, Any],
) -> Dict[str, Any]:
    problem_kernels = kernel_data.get("kernels", {}).get(problem_type, [])
    if not problem_kernels:
        return {
            "atom_infos": [],
            "combination": "",
            "frequency": None,
            "matched_by": "no_problem_type_kernels",
        }

    observed_counts: Dict[str, int] = {}
    observed_original: Dict[str, str] = {}
    for atomic_info in atomic_information:
        atom_text = atomic_info.get("atom_info")
        if not atom_text or atom_text == "No matching generic description found":
            continue
        norm = normalize_atom_info_text(atom_text)
        observed_counts[norm] = observed_counts.get(norm, 0) + 1
        observed_original.setdefault(norm, atom_text)

    kernel_atoms = []
    seen = set()
    for kernel_atom in problem_kernels:
        norm = normalize_atom_info_text(kernel_atom)
        if norm in observed_counts and norm not in seen:
            seen.add(norm)
            kernel_atoms.append(observed_original.get(norm, kernel_atom))

    combination = ", ".join(kernel_atoms)
    frequency = find_kernel_frequency(problem_type, kernel_atoms, kernel_data)

    return {
        "atom_infos": kernel_atoms,
        "combination": combination,
        "frequency": frequency,
        "matched_by": "kernel_intersection",
    }


def find_kernel_frequency(
    problem_type: str,
    kernel_atoms: Sequence[str],
    kernel_data: Dict[str, Any],
) -> Optional[float]:
    if not kernel_atoms:
        return None

    target = sorted(normalize_atom_info_text(atom) for atom in kernel_atoms)
    combinations = kernel_data.get("combinations_frequency", {}).get(problem_type, {})
    for combo_key, frequency in combinations.items():
        combo_atoms = split_kernel_combo(combo_key)
        combo_unique = sorted(set(normalize_atom_info_text(atom) for atom in combo_atoms))
        if combo_unique == target:
            return frequency
    return None


def calculate_problem_overlap(atomic_information: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    distribution: Dict[str, int] = {}
    total_occurrences = 0
    for atomic_info in atomic_information:
        atom_info = atomic_info.get("atom_info")
        if not atom_info or atom_info == "No matching generic description found":
            continue
        total_occurrences += 1
        distribution[atom_info] = distribution.get(atom_info, 0) + 1

    type_count = len(distribution)
    overlap = total_occurrences / type_count if type_count else 0.0
    return {
        "total_atom_info_occurrences": total_occurrences,
        "atom_info_type_count": type_count,
        "overlap": round(overlap, 6),
        "atom_info_distribution": distribution,
    }


def calculate_dataset_statistics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successful_results = [result for result in results if "error" not in result]
    problem_count = len(successful_results)

    problem_type_distribution: Dict[str, int] = {}
    atom_info_distribution: Dict[str, int] = {}
    constraint_type_distribution: Dict[str, int] = {}
    kernel_distribution: Dict[str, int] = {}

    total_atom_info_occurrences = 0
    total_constraint_occurrences = 0
    total_difficulty = 0.0

    for result in successful_results:
        problem_type = result.get("problem_type", "Unknown")
        problem_type_distribution[problem_type] = problem_type_distribution.get(problem_type, 0) + 1
        total_difficulty += float(result.get("difficulty", 0.0) or 0.0)

        kernel_combination = result.get("kernel", {}).get("combination")
        if kernel_combination:
            kernel_distribution[kernel_combination] = kernel_distribution.get(kernel_combination, 0) + 1

        for atomic_info in result.get("atomic_information", []):
            atom_info = atomic_info.get("atom_info")
            if atom_info and atom_info != "No matching generic description found":
                total_atom_info_occurrences += 1
                atom_info_distribution[atom_info] = atom_info_distribution.get(atom_info, 0) + 1

        for constraint in result.get("constraints", []):
            constraint_type = constraint.get("type")
            if constraint_type and constraint_type != "Unclassified":
                total_constraint_occurrences += 1
                constraint_type_distribution[constraint_type] = (
                    constraint_type_distribution.get(constraint_type, 0) + 1
                )

    atom_info_type_count = len(atom_info_distribution)
    overlap = (
        total_atom_info_occurrences / atom_info_type_count
        if atom_info_type_count
        else 0.0
    )
    average_difficulty = total_difficulty / problem_count if problem_count else 0.0

    return {
        "problem_count": problem_count,
        "error_count": len(results) - problem_count,
        "problem_type_count": len(problem_type_distribution),
        "problem_type_distribution": problem_type_distribution,
        "total_atom_info_occurrences": total_atom_info_occurrences,
        "atom_info_type_count": atom_info_type_count,
        "atom_info_distribution": atom_info_distribution,
        "total_constraint_type_occurrences": total_constraint_occurrences,
        "constraint_type_count": len(constraint_type_distribution),
        "constraint_type_distribution": constraint_type_distribution,
        "kernel_combination_count": len(kernel_distribution),
        "kernel_distribution": kernel_distribution,
        "overlap": round(overlap, 6),
        "dataset_overlap": round(overlap, 6),
        "average_difficulty": round(average_difficulty, 6),
    }


def load_constraint_difficulty_map(
    constraint_catalog: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    difficulty_map = {}
    for item in constraint_catalog:
        constraint_type = item.get("type")
        if not constraint_type:
            continue
        difficulty_map[constraint_type] = parse_constraint_base_difficulty(item)
    return difficulty_map


def parse_constraint_base_difficulty(item: Dict[str, Any]) -> float:
    if isinstance(item.get("difficulty_scores"), dict):
        score = item["difficulty_scores"].get("difficulty_score")
        if isinstance(score, (int, float)):
            return float(score)

    if isinstance(item.get("difficulty_score"), (int, float)):
        return float(item["difficulty_score"])

    difficulty = item.get("difficulty", {})
    if isinstance(difficulty, dict):
        return calculate_fallback_difficulty_score(difficulty)

    return 0.0


def calculate_fallback_difficulty_score(difficulty: Dict[str, Any]) -> float:
    aux_var = str(difficulty.get("auxiliary_variables", "No")).lower()
    aux_score = 1 if aux_var == "yes" else 0

    adv_tech = str(difficulty.get("advanced_techniques", "No")).lower()
    adv_score = 1 if adv_tech == "yes" else 0

    var_count = difficulty.get("variable_types_count", 1)
    try:
        var_count = int(var_count)
    except Exception:
        var_count = 1
    var_score_map = {1: 0, 2: 0.33, 3: 0.67, 4: 1}
    var_score = var_score_map.get(var_count, 0)

    return 0.25 * aux_score + 0.35 * adv_score + 0.15 * var_score


def load_constraint_catalogs(
    constraint_catalog_file: Path,
    fallback_constraint_catalog_file: Optional[Path] = FALLBACK_CONSTRAINT_CATALOG_FILE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    primary_catalog = load_json(constraint_catalog_file)
    fallback_catalog: List[Dict[str, Any]] = []
    if fallback_constraint_catalog_file and fallback_constraint_catalog_file.exists():
        fallback_catalog = load_json(fallback_constraint_catalog_file)
    return primary_catalog, fallback_catalog


def build_constraint_difficulty_map(
    primary_catalog: Sequence[Dict[str, Any]],
    fallback_catalog: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    difficulty_map = load_constraint_difficulty_map(primary_catalog)
    for item in fallback_catalog:
        constraint_type = item.get("type")
        if constraint_type and constraint_type not in difficulty_map:
            difficulty_map[constraint_type] = parse_constraint_base_difficulty(item)
    return difficulty_map


def merge_constraint_catalogs_for_prompt(
    primary_catalog: Sequence[Dict[str, Any]],
    fallback_catalog: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = []
    seen = set()
    for catalog in [primary_catalog, fallback_catalog]:
        for item in catalog:
            constraint_type = item.get("type")
            if not constraint_type or constraint_type in seen:
                continue
            seen.add(constraint_type)
            merged.append(item)
    return merged


def calculate_constraint_difficulty(
    constraint: Dict[str, Any],
    constraint_difficulty_map: Dict[str, float],
) -> float:
    base_difficulty = constraint_difficulty_map.get(constraint.get("type", ""), 0.0)
    dimension_difficulty = (
        1.0 if constraint.get("dimensionality") == "multi-dimensional" else 0.0
    )
    return round(base_difficulty + 0.25 * dimension_difficulty, 4)


def calculate_difficulties(
    atomic_information: Sequence[Dict[str, Any]],
    constraint_difficulty_map: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], float]:
    enriched_items = []
    atomic_difficulties = []

    for atomic_info in atomic_information:
        enriched = dict(atomic_info)
        constraints_with_types = []
        total = 0.0
        for constraint in atomic_info.get("constraints_with_types", []):
            enriched_constraint = dict(constraint)
            difficulty = calculate_constraint_difficulty(
                enriched_constraint,
                constraint_difficulty_map,
            )
            enriched_constraint["difficulty"] = difficulty
            total += difficulty
            constraints_with_types.append(enriched_constraint)

        total = round(total, 4)
        enriched["constraints_with_types"] = constraints_with_types
        enriched["difficulty"] = total
        atomic_difficulties.append(total)
        enriched_items.append(enriched)

    return enriched_items, round(max(atomic_difficulties) if atomic_difficulties else 0.0, 4)


def attach_constraint_difficulties(
    constraints: Sequence[Dict[str, Any]],
    constraint_difficulty_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    enriched_constraints = []
    for constraint in constraints:
        enriched = dict(constraint)
        enriched["difficulty"] = calculate_constraint_difficulty(
            enriched,
            constraint_difficulty_map,
        )
        enriched_constraints.append(enriched)
    return enriched_constraints


def load_analysis_resources(
    problem_types_file: Path = DEFAULT_PROBLEM_TYPES_FILE,
    constraint_catalog_file: Path = DEFAULT_CONSTRAINT_CATALOG_FILE,
    atomic_groups_file: Path = DEFAULT_ATOMIC_GROUPS_FILE,
    fallback_constraint_catalog_file: Optional[Path] = FALLBACK_CONSTRAINT_CATALOG_FILE,
    kernel_combinations_file: Path = DEFAULT_KERNEL_COMBINATIONS_FILE,
) -> Dict[str, Any]:
    primary_catalog, fallback_catalog = load_constraint_catalogs(
        constraint_catalog_file,
        fallback_constraint_catalog_file,
    )
    constraint_catalog = merge_constraint_catalogs_for_prompt(
        primary_catalog,
        fallback_catalog,
    )
    return {
        "problem_types": load_json(problem_types_file),
        "constraint_catalog": constraint_catalog,
        "atomic_group_map": load_atomic_group_map(atomic_groups_file),
        "kernel_data": load_kernel_data(kernel_combinations_file),
        "constraint_difficulty_map": build_constraint_difficulty_map(
            primary_catalog,
            fallback_catalog,
        ),
    }


def analyze_problem_difficulty_with_resources(
    question: str,
    modeling: Optional[str],
    *,
    client: Any,
    model: str,
    resources: Dict[str, Any],
) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    resolved_modeling = modeling or generate_modeling(question, client, model)
    atomic_information = extract_atomic_information(
        question,
        resolved_modeling,
        client,
        model,
    )
    problem_type = classify_problem_type(
        question,
        resources["problem_types"],
        client,
        model,
    )

    constraints, constraint_sources = unique_constraint_sources(atomic_information)
    classified_constraints = classify_constraints(
        constraints,
        problem_type,
        resources["constraint_catalog"],
        client,
        model,
        constraint_sources=constraint_sources,
    )
    atomic_information = attach_constraint_types_to_atomic_information(
        atomic_information,
        classified_constraints,
    )
    atomic_information = attach_atom_info(
        atomic_information,
        problem_type,
        resources["atomic_group_map"],
    )
    kernel = get_problem_kernel(
        atomic_information,
        problem_type,
        resources["kernel_data"],
    )
    atomic_information, problem_difficulty = calculate_difficulties(
        atomic_information,
        resources["constraint_difficulty_map"],
    )
    overlap = calculate_problem_overlap(atomic_information)
    constraints_with_difficulty = attach_constraint_difficulties(
        list(classified_constraints.values()),
        resources["constraint_difficulty_map"],
    )

    return {
        "question": question,
        "modeling": resolved_modeling,
        "problem_type": problem_type,
        "kernel": kernel,
        "atomic_information": atomic_information,
        "constraints": constraints_with_difficulty,
        "overlap": overlap,
        "difficulty": problem_difficulty,
        "difficulty_rule": (
            "constraint difficulty = base type difficulty + 0.25 if multi-dimensional; "
            "atomic difficulty = sum of its related constraints; "
            "problem difficulty = max atomic difficulty"
        ),
    }


def analyze_problem_difficulty(
    question: str,
    modeling: Optional[str] = None,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    problem_types_file: Path = DEFAULT_PROBLEM_TYPES_FILE,
    constraint_catalog_file: Path = DEFAULT_CONSTRAINT_CATALOG_FILE,
    atomic_groups_file: Path = DEFAULT_ATOMIC_GROUPS_FILE,
    fallback_constraint_catalog_file: Optional[Path] = FALLBACK_CONSTRAINT_CATALOG_FILE,
    kernel_combinations_file: Path = DEFAULT_KERNEL_COMBINATIONS_FILE,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze one optimization problem and return a structured difficulty report.

    Args:
        question: Natural-language optimization problem.
        modeling: Optional mathematical model. If omitted, the interface calls
            the LLM to generate one first.
        client: Optional OpenAI-compatible client.
        model: LLM model name.
        *_file: Metadata file paths.
        api_key/base_url: Optional OpenAI-compatible API settings.
    """
    llm_client = client or create_client(api_key=api_key, base_url=base_url)
    resources = load_analysis_resources(
        problem_types_file=problem_types_file,
        constraint_catalog_file=constraint_catalog_file,
        atomic_groups_file=atomic_groups_file,
        fallback_constraint_catalog_file=fallback_constraint_catalog_file,
        kernel_combinations_file=kernel_combinations_file,
    )
    return analyze_problem_difficulty_with_resources(
        question,
        modeling,
        client=llm_client,
        model=model,
        resources=resources,
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_no} must be an object")
            records.append(record)
    return records


def write_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_present(record: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_question_and_modeling(record: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    question = first_present(
        record,
        [
            "question",
            "problem",
            "problem_text",
            "problem_description",
            "natural_language_desc",
        ],
    )
    modeling = first_present(record, ["modeling", "abstract_modeling", "model"])

    generation = record.get("generation")
    if isinstance(generation, dict):
        question = question or first_present(
            generation,
            ["natural_language_desc", "problem_description"],
        )
        modeling = modeling or first_present(generation, ["abstract_modeling", "modeling"])

    if not question:
        raise ValueError(
            "JSONL record is missing a question field. Supported fields include "
            "question, problem, problem_text, problem_description, natural_language_desc, "
            "and generation.problem_description."
        )
    return question, modeling


def source_record_id(record: Dict[str, Any], fallback_index: int) -> Any:
    for key in ["idx", "id", "uid", "problem_id", "task_id"]:
        if key in record:
            return record[key]
    return fallback_index


def analyze_problem_records(
    records: Sequence[Dict[str, Any]],
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    problem_types_file: Path = DEFAULT_PROBLEM_TYPES_FILE,
    constraint_catalog_file: Path = DEFAULT_CONSTRAINT_CATALOG_FILE,
    atomic_groups_file: Path = DEFAULT_ATOMIC_GROUPS_FILE,
    fallback_constraint_catalog_file: Optional[Path] = FALLBACK_CONSTRAINT_CATALOG_FILE,
    kernel_combinations_file: Path = DEFAULT_KERNEL_COMBINATIONS_FILE,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    output_jsonl: Optional[Path] = None,
    start_index: int = 0,
    limit: Optional[int] = None,
    continue_on_error: bool = True,
    include_source_metadata: bool = False,
) -> List[Dict[str, Any]]:
    llm_client = client or create_client(api_key=api_key, base_url=base_url)
    resources = load_analysis_resources(
        problem_types_file=problem_types_file,
        constraint_catalog_file=constraint_catalog_file,
        atomic_groups_file=atomic_groups_file,
        fallback_constraint_catalog_file=fallback_constraint_catalog_file,
        kernel_combinations_file=kernel_combinations_file,
    )

    selected_records = list(records[start_index:])
    if limit is not None:
        selected_records = selected_records[:limit]

    results = []
    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text("", encoding="utf-8")

    for offset, record in enumerate(selected_records):
        input_index = start_index + offset
        result: Dict[str, Any]
        try:
            question, modeling = extract_question_and_modeling(record)
            result = analyze_problem_difficulty_with_resources(
                question,
                modeling,
                client=llm_client,
                model=model,
                resources=resources,
            )
            if include_source_metadata:
                result["input_index"] = input_index
                result["source_id"] = source_record_id(record, input_index)
        except Exception as exc:
            if not continue_on_error:
                raise
            result = {
                "error": str(exc),
            }
            if include_source_metadata:
                result["input_index"] = input_index
                result["source_id"] = source_record_id(record, input_index)

        results.append(result)
        if output_jsonl:
            write_jsonl_record(output_jsonl, result)

    return results


def analyze_jsonl_file(
    input_jsonl: Path,
    output_jsonl: Optional[Path] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    records = read_jsonl(input_jsonl)
    return analyze_problem_records(records, output_jsonl=output_jsonl, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the difficulty of one optimization problem or a JSONL file of problems."
    )
    parser.add_argument("--question", help="Problem text. If omitted, stdin is used.")
    parser.add_argument("--question-file", type=Path, help="Read problem text from a file.")
    parser.add_argument("--input-jsonl", type=Path, help="Read multiple problem records from JSONL.")
    parser.add_argument("--modeling", help="Optional existing mathematical modeling text.")
    parser.add_argument("--modeling-file", type=Path, help="Read existing modeling from a file.")
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    parser.add_argument("--output-jsonl", type=Path, help="Optional JSONL output file for batch mode.")
    parser.add_argument("--stats-output", type=Path, help="Optional dataset statistics JSON output file for batch mode.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for JSONL batch mode.")
    parser.add_argument("--limit", type=int, help="Maximum number of JSONL records to process.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop JSONL batch mode on the first failed record.",
    )
    parser.add_argument(
        "--include-source-metadata",
        action="store_true",
        help="Include input_index/source_id in JSONL batch results.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI-compatible model name.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--problem-types-file",
        type=Path,
        default=DEFAULT_PROBLEM_TYPES_FILE,
    )
    parser.add_argument(
        "--constraint-catalog-file",
        type=Path,
        default=DEFAULT_CONSTRAINT_CATALOG_FILE,
    )
    parser.add_argument(
        "--fallback-constraint-catalog-file",
        type=Path,
        default=FALLBACK_CONSTRAINT_CATALOG_FILE,
        help="Fallback constraint catalog used when a type is missing from the score table.",
    )
    parser.add_argument(
        "--atomic-groups-file",
        type=Path,
        default=DEFAULT_ATOMIC_GROUPS_FILE,
    )
    parser.add_argument(
        "--kernel-combinations-file",
        type=Path,
        default=DEFAULT_KERNEL_COMBINATIONS_FILE,
    )
    return parser


def read_input_text(value: Optional[str], path: Optional[Path]) -> Optional[str]:
    if path:
        return path.read_text(encoding="utf-8")
    return value


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    batch_input = args.input_jsonl
    if (
        batch_input is None
        and args.question_file
        and args.question_file.suffix.lower() == ".jsonl"
        and not args.question
        and not args.modeling
        and not args.modeling_file
    ):
        batch_input = args.question_file

    if batch_input:
        output_jsonl = args.output_jsonl or args.output
        results = analyze_jsonl_file(
            batch_input,
            output_jsonl=output_jsonl,
            model=args.model,
            problem_types_file=args.problem_types_file,
            constraint_catalog_file=args.constraint_catalog_file,
            atomic_groups_file=args.atomic_groups_file,
            fallback_constraint_catalog_file=args.fallback_constraint_catalog_file,
            kernel_combinations_file=args.kernel_combinations_file,
            api_key=args.api_key,
            base_url=args.base_url,
            start_index=args.start_index,
            limit=args.limit,
            continue_on_error=not args.stop_on_error,
            include_source_metadata=args.include_source_metadata,
        )
        statistics = calculate_dataset_statistics(results)
        stats_output = args.stats_output
        if output_jsonl and stats_output is None:
            stats_output = Path(f"{output_jsonl}.stats.json")
        if stats_output:
            write_json(stats_output, statistics)
        if not output_jsonl:
            print(json.dumps({"results": results, "statistics": statistics}, ensure_ascii=False, indent=2))
        return

    question = read_input_text(args.question, args.question_file)
    if not question:
        question = input().strip()
    modeling = read_input_text(args.modeling, args.modeling_file)

    result = analyze_problem_difficulty(
        question,
        modeling,
        model=args.model,
        problem_types_file=args.problem_types_file,
        constraint_catalog_file=args.constraint_catalog_file,
        atomic_groups_file=args.atomic_groups_file,
        fallback_constraint_catalog_file=args.fallback_constraint_catalog_file,
        kernel_combinations_file=args.kernel_combinations_file,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
