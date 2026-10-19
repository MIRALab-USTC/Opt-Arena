#!/usr/bin/env python3
"""
Problem generation interface for Opt-Arena.

Modes:
1. manual: generate by requested problem type counts.
2. similar: read analyzed difficulty JSONL, aggregate by (problem_type, kernel),
   and generate similar problems from those kernels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


EVO_OPT_ARENA_DIR = Path(os.getenv("EVO_OPT_ARENA_DIR", str(Path(__file__).resolve().parent)))
MANUAL_GENERATOR_FILE = EVO_OPT_ARENA_DIR / "generate_problem_manual_mode.py"
DEFAULT_PROBLEM_TYPES_FILE = EVO_OPT_ARENA_DIR / "problem_type.json"
DEFAULT_PT_STATS_FILE = EVO_OPT_ARENA_DIR / "problem_types_atomic_info_statistics.json"
DEFAULT_KERNELS_FILE = EVO_OPT_ARENA_DIR / "kernel_combinations.json"
DEFAULT_ATOMIC_ALL_FILE = EVO_OPT_ARENA_DIR / "atomic_info_all-2.json"
DEFAULT_CONSTRAINTS_FILE = EVO_OPT_ARENA_DIR / "constraints_with_difficulty_scores-train.json"
DEFAULT_OUTPUT_JSONL = Path("outputs/generated_problems.jsonl")

DEFAULT_MODEL = os.getenv("OPT_ARENA_MODEL") or os.getenv("DEEPSEEK_MODEL") or "gpt-4o-mini"
DEFAULT_BASE_URL = (
    os.getenv("OPT_ARENA_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
)

TEMPERATURE_CONFIG = {
    "problem": 0.7,
    "modeling": 0.3,
    "code": 0.2,
    "nl": 0.5,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\t\n\r]", " ", text)
    text = re.sub(r"[\.,;:!\?\(\)\[\]\{\}\"']", "", text)
    return text.strip()


def normalize_atom(text: Any) -> str:
    return " ".join(str(text).strip().rstrip(".").split()).lower()


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    resolved_key = (
        api_key
        or os.getenv("OPT_ARENA_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY.")
    kwargs = {"api_key": resolved_key}
    resolved_base_url = base_url or DEFAULT_BASE_URL
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def call_generate(
    client: Any,
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int = 4000,
    retries: int = 3,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert operations research problem generator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2)
    raise RuntimeError(f"LLM generation failed: {last_error}") from last_error


def load_prompt_module(path: Path = MANUAL_GENERATOR_FILE) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt module not found: {path}. Supply --prompt-module-file.")
    spec = importlib.util.spec_from_file_location("opt_arena_manual_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompt module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_kernel_combo(combo_key: str) -> List[str]:
    parts = []
    chunks = combo_key.split(".,") if ".," in combo_key else [combo_key]
    for chunk in chunks:
        cleaned = chunk.strip().rstrip(".").strip()
        if cleaned:
            parts.append(f"{cleaned}.")
    return parts


def load_resources(
    problem_types_file: Path = DEFAULT_PROBLEM_TYPES_FILE,
    pt_stats_file: Path = DEFAULT_PT_STATS_FILE,
    kernels_file: Path = DEFAULT_KERNELS_FILE,
    atomic_all_file: Path = DEFAULT_ATOMIC_ALL_FILE,
    constraints_file: Path = DEFAULT_CONSTRAINTS_FILE,
) -> Dict[str, Any]:
    problem_types = load_json(problem_types_file)
    pt_stats = load_json(pt_stats_file) if pt_stats_file.exists() else {"statistics": []}
    kernels = load_json(kernels_file)
    atomic_all = load_json(atomic_all_file)
    constraints = load_json(constraints_file)

    atomic_pool_by_type: Dict[str, List[str]] = {}
    for entry in pt_stats.get("statistics", []):
        ptype = entry.get("problem_type")
        atoms = [a for a in entry.get("atomic_infos", []) if isinstance(a, str) and a.strip()]
        if ptype and atoms:
            atomic_pool_by_type[ptype] = list(dict.fromkeys(atoms))

    atomic_index: Dict[str, Dict[str, List[str]]] = {}
    for item in atomic_all:
        ptype = item.get("problem_type")
        merged_info = item.get("merged_info")
        if not ptype or not merged_info:
            continue
        atomic_pool_by_type.setdefault(ptype, [])
        if merged_info not in atomic_pool_by_type[ptype]:
            atomic_pool_by_type[ptype].append(merged_info)
        key = normalize_atom(merged_info)
        atomic_index.setdefault(ptype, {})
        existing = atomic_index[ptype].get(key, [])
        atomic_index[ptype][key] = sorted(set(existing + item.get("constraint_types", [])))

    constraint_catalog = {item.get("type"): item for item in constraints if item.get("type")}
    known_types = {item.get("problem") for item in problem_types if item.get("problem")}
    known_types.update(atomic_pool_by_type.keys())
    known_types.update(kernels.get("kernels", {}).keys())
    known_types.update(kernels.get("combinations_frequency", {}).keys())

    return {
        "problem_types": problem_types,
        "pt_stats": pt_stats,
        "kernels": kernels,
        "atomic_all": atomic_all,
        "atomic_pool_by_type": atomic_pool_by_type,
        "atomic_index": atomic_index,
        "constraint_catalog": constraint_catalog,
        "known_types": known_types,
    }


def parse_type_counts(value: Optional[str], path: Optional[Path]) -> Dict[str, int]:
    if path:
        data = load_json(path)
    elif value:
        data = json.loads(value)
    else:
        raise ValueError("manual mode requires --type-counts or --type-counts-file")
    if not isinstance(data, dict):
        raise ValueError("type counts must be a JSON object: {problem_type: count}")
    return {str(k): int(v) for k, v in data.items() if int(v) > 0}


def sample_kernel(problem_type: str, resources: Dict[str, Any]) -> List[str]:
    combinations = resources["kernels"].get("combinations_frequency", {}).get(problem_type)
    if combinations:
        choices = list(combinations.keys())
        weights = list(combinations.values())
        return split_kernel_combo(random.choices(choices, weights=weights, k=1)[0])

    kernels = resources["kernels"].get("kernels", {}).get(problem_type, [])
    if kernels:
        return random.sample(kernels, min(3, len(kernels)))

    pool = resources["atomic_pool_by_type"].get(problem_type, [])
    return random.sample(pool, min(3, len(pool))) if pool else []


def supplement_atomics(
    problem_type: str,
    seed_atomics: Sequence[str],
    resources: Dict[str, Any],
    target_atomic_count: int,
) -> List[str]:
    selected = []
    seen = set()
    for atom in seed_atomics:
        if not atom:
            continue
        key = normalize_atom(atom)
        if key not in seen:
            selected.append(atom)
            seen.add(key)

    pool = resources["atomic_pool_by_type"].get(problem_type, [])
    candidates = [atom for atom in pool if normalize_atom(atom) not in seen]
    needed = max(target_atomic_count - len(selected), 0)
    if needed and candidates:
        selected.extend(random.sample(candidates, min(needed, len(candidates))))
    return selected


def find_constraint_types_for_atom(problem_type: str, atom: str, resources: Dict[str, Any]) -> List[str]:
    index = resources["atomic_index"].get(problem_type, {})
    norm_atom = normalize_atom(atom)
    if norm_atom in index:
        return index[norm_atom]
    for key, constraint_types in index.items():
        if norm_atom in key or key in norm_atom:
            return constraint_types
    return []


def map_atomic_logic(chosen_atomics: Sequence[str], problem_type: str, resources: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapped = []
    catalog = resources["constraint_catalog"]
    for atom in chosen_atomics:
        constraint_types = find_constraint_types_for_atom(problem_type, atom, resources)
        details = []
        for constraint_type in constraint_types:
            if constraint_type in catalog:
                item = catalog[constraint_type]
                details.append({
                    "type": constraint_type,
                    "description": item.get("description"),
                    "index": item.get("index"),
                })
        mapped.append({"atomic_info": atom, "constraints": details})
    return mapped


def create_manual_tasks(type_counts: Dict[str, int], resources: Dict[str, Any], target_atomic_count: int) -> List[Dict[str, Any]]:
    tasks = []
    for problem_type, count in type_counts.items():
        is_known = problem_type in resources["known_types"]
        for _ in range(count):
            kernel = sample_kernel(problem_type, resources) if is_known else []
            final_atomics = supplement_atomics(problem_type, kernel, resources, target_atomic_count) if is_known else []
            mapped = map_atomic_logic(final_atomics, problem_type, resources) if is_known else []
            tasks.append({
                "problem_type": problem_type,
                "mode": "Mode A (Complex)" if is_known else "Mode B (Direct)",
                "kernel": kernel,
                "selected_atomic_infos": final_atomics,
                "mapped": mapped,
                "source": "manual",
            })
    return tasks


def extract_kernel_atoms(record: Dict[str, Any]) -> List[str]:
    kernel = record.get("kernel")
    if isinstance(kernel, dict):
        atoms = kernel.get("atom_infos")
        if isinstance(atoms, list) and atoms:
            return [str(atom) for atom in atoms if str(atom).strip()]

    distribution = record.get("overlap", {}).get("atom_info_distribution")
    if isinstance(distribution, dict) and distribution:
        return list(distribution.keys())

    atoms = []
    for atomic_info in record.get("atomic_information", []):
        atom = atomic_info.get("atom_info")
        if atom and atom != "No matching generic description found":
            atoms.append(atom)
    return list(dict.fromkeys(atoms))


def create_similar_tasks_from_analysis(
    analysis_records: Sequence[Dict[str, Any]],
    resources: Dict[str, Any],
    similar_count: int,
    target_atomic_count: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    for record in analysis_records:
        if "error" in record:
            continue
        problem_type = record.get("problem_type")
        if not problem_type:
            continue
        kernel_atoms = extract_kernel_atoms(record)
        key = (problem_type, tuple(sorted(dict.fromkeys(kernel_atoms))))
        grouped[key] = grouped.get(key, 0) + 1

    tasks = []
    for (problem_type, kernel_tuple), observed_count in grouped.items():
        for _ in range(similar_count):
            kernel = list(kernel_tuple)
            final_atomics = supplement_atomics(problem_type, kernel, resources, target_atomic_count)
            mapped = map_atomic_logic(final_atomics, problem_type, resources)
            tasks.append({
                "problem_type": problem_type,
                "mode": "Mode A (Similar Kernel)",
                "kernel": kernel,
                "selected_atomic_infos": final_atomics,
                "mapped": mapped,
                "source": "similar",
                "observed_count": observed_count,
            })
    return tasks


def create_similar_tasks_from_plan(
    plan_file: Path,
    resources: Dict[str, Any],
    target_atomic_count: int,
) -> List[Dict[str, Any]]:
    plan = load_json(plan_file)
    if not isinstance(plan, list):
        raise ValueError("similar plan file must contain a JSON list")

    tasks = []
    for item in plan:
        problem_type = item.get("problem_type")
        count = int(item.get("count", 1))
        kernel = item.get("kernel") or item.get("kernel_atom_infos") or item.get("atom_infos") or []
        if isinstance(kernel, str):
            kernel = split_kernel_combo(kernel)
        if not problem_type or count <= 0:
            continue
        for _ in range(count):
            final_atomics = supplement_atomics(problem_type, kernel, resources, target_atomic_count)
            mapped = map_atomic_logic(final_atomics, problem_type, resources)
            tasks.append({
                "problem_type": problem_type,
                "mode": "Mode A (Similar Kernel)",
                "kernel": list(kernel),
                "selected_atomic_infos": final_atomics,
                "mapped": mapped,
                "source": "similar_plan",
            })
    return tasks


def generate_one_task(
    task: Dict[str, Any],
    *,
    client: Any,
    model: str,
    prompt_module: Any,
    include_code: bool = True,
    include_nl: bool = True,
) -> Dict[str, Any]:
    problem_type = task["problem_type"]
    mode = task["mode"]
    mapped = task.get("mapped", [])
    if mode == "Mode B (Direct)":
        prompts = prompt_module.get_prompts_direct(problem_type)
    else:
        prompts = prompt_module.get_prompts_complex(problem_type, mapped)

    problem_text = call_generate(
        client,
        model,
        prompts["p"],
        temperature=TEMPERATURE_CONFIG["problem"],
        max_tokens=4000,
    )

    modeling_text = call_generate(
        client,
        model,
        prompts["m"].format(problem_text=problem_text),
        temperature=TEMPERATURE_CONFIG["modeling"],
        max_tokens=4000,
    )

    code_text = None
    if include_code:
        code_text = call_generate(
            client,
            model,
            prompts["c"].format(problem_text=problem_text, modeling_text=modeling_text),
            temperature=TEMPERATURE_CONFIG["code"],
            max_tokens=4000,
        )

    natural_language_desc = None
    if include_nl:
        natural_language_desc = call_generate(
            client,
            model,
            prompts["nl"].format(problem_text=problem_text),
            temperature=TEMPERATURE_CONFIG["nl"],
            max_tokens=4000,
        )

    return {
        "problem_type": problem_type,
        "mode": mode,
        "kernel": task.get("kernel", []),
        "selected_atomic_infos": task.get("selected_atomic_infos", []),
        "source": task.get("source"),
        "observed_count": task.get("observed_count"),
        "generation": {
            "problem_description": problem_text,
            "abstract_modeling": modeling_text,
            "gurobi_code": code_text,
            "natural_language_desc": natural_language_desc,
            "timestamp": int(time.time()),
        },
    }


def generate_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    output_jsonl: Path,
    client: Any,
    model: str,
    prompt_module: Any,
    max_workers: int = 1,
    include_code: bool = True,
    include_nl: bool = True,
) -> List[Dict[str, Any]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")
    results = []

    def run(task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return generate_one_task(
                task,
                client=client,
                model=model,
                prompt_module=prompt_module,
                include_code=include_code,
                include_nl=include_nl,
            )
        except Exception as exc:
            return {
                "problem_type": task.get("problem_type"),
                "mode": task.get("mode"),
                "kernel": task.get("kernel", []),
                "selected_atomic_infos": task.get("selected_atomic_infos", []),
                "source": task.get("source"),
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run, task): task for task in tasks}
        for future in as_completed(future_to_task):
            record = future.result()
            results.append(record)
            write_jsonl(output_jsonl, [record], append=True)
            print(f"[generated] {len(results)}/{len(tasks)} {record.get('problem_type')} {record.get('mode')}")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Opt-Arena problems.")
    parser.add_argument("--mode", choices=["manual", "similar"], required=True)
    parser.add_argument("--type-counts", help='Manual mode JSON, e.g. {"Production Planning Problem": 2}')
    parser.add_argument("--type-counts-file", type=Path)
    parser.add_argument("--analysis-jsonl", type=Path, help="Difficulty analysis JSONL for similar mode.")
    parser.add_argument("--similar-count", type=int, default=1, help="Problems to generate per observed (problem_type, kernel).")
    parser.add_argument("--similar-plan-file", type=Path, help="Optional explicit similar generation plan JSON list.")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--target-atomic-count", type=int, default=6)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-nl", action="store_true")
    parser.add_argument("--problem-types-file", type=Path, default=DEFAULT_PROBLEM_TYPES_FILE)
    parser.add_argument("--pt-stats-file", type=Path, default=DEFAULT_PT_STATS_FILE)
    parser.add_argument("--kernels-file", type=Path, default=DEFAULT_KERNELS_FILE)
    parser.add_argument("--atomic-all-file", type=Path, default=DEFAULT_ATOMIC_ALL_FILE)
    parser.add_argument("--constraints-file", type=Path, default=DEFAULT_CONSTRAINTS_FILE)
    parser.add_argument("--prompt-module-file", type=Path, default=MANUAL_GENERATOR_FILE)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    resources = load_resources(
        problem_types_file=args.problem_types_file,
        pt_stats_file=args.pt_stats_file,
        kernels_file=args.kernels_file,
        atomic_all_file=args.atomic_all_file,
        constraints_file=args.constraints_file,
    )

    if args.mode == "manual":
        type_counts = parse_type_counts(args.type_counts, args.type_counts_file)
        tasks = create_manual_tasks(type_counts, resources, args.target_atomic_count)
    elif args.similar_plan_file:
        tasks = create_similar_tasks_from_plan(args.similar_plan_file, resources, args.target_atomic_count)
    else:
        if not args.analysis_jsonl:
            raise ValueError("similar mode requires --analysis-jsonl or --similar-plan-file")
        analysis_records = read_jsonl(args.analysis_jsonl)
        tasks = create_similar_tasks_from_analysis(
            analysis_records,
            resources,
            args.similar_count,
            args.target_atomic_count,
        )

    if not tasks:
        raise RuntimeError("No generation tasks were created.")

    print(f"Prepared {len(tasks)} generation tasks")
    prompt_module = load_prompt_module(args.prompt_module_file)
    client = create_client(api_key=args.api_key, base_url=args.base_url)
    generate_tasks(
        tasks,
        output_jsonl=args.output_jsonl,
        client=client,
        model=args.model,
        prompt_module=prompt_module,
        max_workers=args.max_workers,
        include_code=not args.skip_code,
        include_nl=not args.skip_nl,
    )
    print(f"Saved generated problems to {args.output_jsonl}")


if __name__ == "__main__":
    main()
