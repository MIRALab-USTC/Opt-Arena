import os
import json
import math
import random
import time
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ================= Dependency checks =================
try:
    import ijson
except Exception:
    ijson = None
    print("[WARN] ijson not found. Streaming large JSONs might fail.")

try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None
    print("[WARN] openai not found. Generation will fail.")

# ================= Utility functions =================

def load_json(path: str) -> Any:
    """Load a JSON file safely"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(s: str) -> str:
    """Normalize text for matching"""
    import re
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\t\n\r]", " ", s)
    s = re.sub(r"[\.,;:!\?\(\)\[\]\{\}\"']", "", s)
    return s

def ensure_dir(path: str) -> None:
    """Ensure a directory exists"""
    os.makedirs(path, exist_ok=True)

def get_client_from_env() -> Optional[Any]:
    """Create an API client from environment variables"""
    api_key = os.getenv("OPT_ARENA_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPT_ARENA_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key: 
        return None
    return OpenAI(api_key=api_key, base_url=base_url)

def call_deepseek_generate(client: Any, model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Generate content with the DeepSeek API and retry on failure"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert operations research problem generator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"[ERROR] API Call failed: {e}")
            return ""

# ================= Core logic: generation plan =================

def calculate_generation_plan(stats_file: str, total_target: int) -> List[Dict]:
    """
    Calculate the number of samples per type from statistics (automatic mode)
    """
    print(f"Building generation plan for {total_target} samples...")
    
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistics file not found: {stats_file}. Run the statistics script first.")

    with open(stats_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Aggregate data by problem name across datasets
    type_stats = {}
    
    for dataset_entry in data:
        for p_type_data in dataset_entry.get("problem_types", []):
            p_name = p_type_data["problem_type"]
            total = p_type_data["total"]
            correct = p_type_data["correct"]
            
            if p_name not in type_stats:
                type_stats[p_name] = {"total": 0, "correct": 0}
            type_stats[p_name]["total"] += total
            type_stats[p_name]["correct"] += correct

    # 2. Calculate weights
    plan_list = []
    for p_name, stats in type_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0.0
        
        # Error-driven weight formula: (1 - accuracy)^1.5 + 0.1
        error_rate = 1.0 - acc
        weight = (error_rate ** 1.5) + 0.1
        
        plan_list.append({
            "problem_type": p_name,
            "accuracy": acc,
            "weight": weight
        })

    # Normalize weights and calculate counts
    df = pd.DataFrame(plan_list)
    total_weight = df['weight'].sum()
    df['norm_weight'] = df['weight'] / total_weight
    
    # Calculate floating-point allocations first
    df['target_samples_float'] = df['norm_weight'] * total_target
    
    # Apply minimum allocations before rounding
    df['target_samples_float'] = df['target_samples_float'].apply(
        lambda x: max(x, float(MIN_SAMPLES_PER_TYPE))
    )
    
    # Round down
    df['target_samples'] = df['target_samples_float'].astype(int)
    
    # Renormalize the total
    current_total = df['target_samples'].sum()
    
    # If the total remains zero, use the minimum allocation
    if current_total == 0:
        df['target_samples'] = MIN_SAMPLES_PER_TYPE
        current_total = df['target_samples'].sum()
    
    # Adjust to the target total
    if current_total > 0 and current_total != total_target:
        scale = total_target / current_total
        df['target_samples'] = (df['target_samples'] * scale).astype(int)
    
    # If still short, add samples to the first types
    final_total = df['target_samples'].sum()
    if final_total < total_target:
        deficit = total_target - final_total
        # Add samples in descending weight order
        sorted_indices = df['norm_weight'].sort_values(ascending=False).index
        for idx in sorted_indices[:deficit]:
            df.loc[idx, 'target_samples'] += 1
    
    # Convert to a list
    final_plan = df[['problem_type', 'target_samples', 'accuracy']].to_dict(orient='records')
    
    # Print a preview
    print("-" * 60)
    print(f"{'Problem Type':<40} | {'Acc':<6} | {'Count':<6}")
    print("-" * 60)
    for item in sorted(final_plan, key=lambda x: x['target_samples'], reverse=True):
        print(f"{item['problem_type']:<40} | {item['accuracy']:.2f}   | {item['target_samples']}")
    print("-" * 60)
    
    return final_plan

def create_manual_plan(manual_distribution: Dict[str, int], pt_stats_path: str) -> List[Dict]:
    """
    Create a generation plan from manual configuration
    """
    print(f"Using a manually configured generation plan...")
    
    # Load statistics to validate problem types
    try:
        pt_stats = load_json(pt_stats_path)
        known_types = set(entry.get("problem_type") for entry in pt_stats.get("statistics", []))
    except Exception as e:
        print(f"[WARN] Could not load statistics: {e}; skipping type validation")
        known_types = set()
    
    manual_plan = []
    print("\n" + "=" * 60)
    print(f"{'Problem Type':<40} | {'Count':<6}")
    print("=" * 60)
    
    for ptype, count in manual_distribution.items():
        # Check whether the type is known
        if known_types and ptype not in known_types:
            print(f"[WARN] Unknown problem type '{ptype}'; falling back to Mode B (Direct)")
        
        # Validate the count
        if not isinstance(count, int) or count <= 0:
            print(f"[WARN] Invalid generation count {count} for {ptype}; skipping")
            continue
        
        manual_plan.append({
            "problem_type": ptype,
            "target_samples": count,
            "mode": "manual"
        })
        print(f"{ptype:<40} | {count}")
    
    print("=" * 60)
    total = sum(item['target_samples'] for item in manual_plan)
    print(f"Total: {total} samples\n")
    
    return manual_plan

# ================= Configuration =================

# 1. Path configuration
ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_STATS_FILE = os.getenv("OPTMATH_EVALUATION_STATS_FILE", os.path.join(ROOT, "data", "problem_type_analysis.json"))  # Statistics file (automatic mode only)
PT_STATS_PATH = os.getenv("OPTMATH_PT_STATS_FILE", os.path.join(ROOT, "data", "problem_types_atomic_info_statistics.json"))
KERNELS_PATH = os.path.join(ROOT, "kernel_combinations.json")
ATOMIC_ALL_PATH = os.path.join(ROOT, "atomic_info_all-2.json")
CONSTRAINTS_CATALOG_PATH = os.path.join(ROOT, "constraints_with_difficulty_scores-train.json")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "generated_problems_manual.jsonl")

# ================= Mode selection =================
# True: Manual mode: specify the count for each problem type
# False: Automatic mode: derive counts from evaluation statistics
USE_MANUAL_MODE = True

# ================= Manual mode configuration =================
# Used when USE_MANUAL_MODE is True
# Format: {"Problem type name": count}
MANUAL_TYPE_DISTRIBUTION = {
    "Minimum Cost Flow Problem": 2,
    "Set Cover Problem": 2,
    "Multiple Knapsack Problem": 2,
    "Lot Sizing Problem": 2,
    "Capacitated Vehicle Routing Problem": 2,
    "Resource-Constrained Project Scheduling": 2,
}

# ================= Automatic mode configuration =================
# Used when USE_MANUAL_MODE is False
TARGET_TOTAL_SAMPLES = 1000   # Target total number of samples
MIN_SAMPLES_PER_TYPE = 10     # Minimum allocation to avoid omitting a type

# ================= Parallel processing configuration =================
MAX_WORKERS = 6               # Maximum concurrent worker threads
BATCH_SIZE = 30               # Tasks per batch

# ================= Temperature configuration =================
# Temperature settings for generation steps
TEMPERATURE_CONFIG = {
    "problem": 0.7,   # Higher creativity for varied problem descriptions
    "modeling": 0.3,  # Lower temperature for rigorous mathematical modeling
    "code": 0.2,      # Very low temperature for code accuracy
    "nl": 0.5         # Moderate temperature for varied natural-language descriptions
}

def generate_single_problem(problem_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate one problem through all four stages"""
    ptype = problem_args["ptype"]
    mapped = problem_args["mapped"]
    kernel = problem_args["kernel"]
    final_atomics = problem_args["final_atomics"]
    model = problem_args["model"]
    client = problem_args["client"]
    mode = problem_args["mode"]
    
    print(f"[Thread-{threading.current_thread().ident}] Generating problem for type: {ptype}")
    
    # --- Step 0: Prepare prompt ---
    prompts_dict = {}
    if mode == "Mode A (Complex)":
        prompts_dict = get_prompts_complex(ptype, mapped)
    else:
        prompts_dict = get_prompts_direct(ptype)
    
    # --- Step 1: Generate Problem ---
    problem_text_raw: Optional[str] = None
    try:
        p_text = call_deepseek_generate(client, model, prompts_dict["p"], 
                                        temperature=TEMPERATURE_CONFIG["problem"], 
                                        max_tokens=4000)
        if not p_text: raise Exception("Empty generation")
        problem_text_raw = p_text
        print(f"[Thread-{threading.current_thread().ident}] Problem generation completed for {ptype}.")
    except Exception as e:
        print(f"[Thread-{threading.current_thread().ident}] [WARN] Problem generation failed for {ptype}: {e}")
        return None

    # --- Step 2: Modeling ---
    modeling_text_raw: Optional[str] = None
    try:
        m_prompt_fmt = prompts_dict["m"].format(problem_text=p_text)
        m_text = call_deepseek_generate(client, model, m_prompt_fmt, 
                                        temperature=TEMPERATURE_CONFIG["modeling"], 
                                        max_tokens=4000)
        modeling_text_raw = m_text
        print(f"[Thread-{threading.current_thread().ident}] Modeling generation completed for {ptype}.")
    except Exception as e:
        print(f"[Thread-{threading.current_thread().ident}] [WARN] Modeling generation failed for {ptype}: {e}")

    # --- Step 3: Code ---
    code_text_raw: Optional[str] = None
    try:
        modeling_to_use = modeling_text_raw if modeling_text_raw else problem_text_raw
        if modeling_to_use:
            c_prompt_fmt = prompts_dict["c"].format(problem_text=p_text, modeling_text=modeling_to_use)
            c_text = call_deepseek_generate(client, model, c_prompt_fmt, 
                                            temperature=TEMPERATURE_CONFIG["code"], 
                                            max_tokens=4000)
            code_text_raw = c_text
        print(f"[Thread-{threading.current_thread().ident}] Code generation completed for {ptype}.")
    except Exception as e:
        print(f"[Thread-{threading.current_thread().ident}] [WARN] Code generation failed for {ptype}: {e}")

    # --- Step 4: NL Desc ---
    natural_language_desc: Optional[str] = None
    try:
        nl_prompt_fmt = prompts_dict["nl"].format(problem_text=p_text)
        nl_text = call_deepseek_generate(client, model, nl_prompt_fmt, 
                                         temperature=TEMPERATURE_CONFIG["nl"], 
                                         max_tokens=4000)
        natural_language_desc = nl_text
        print(f"[Thread-{threading.current_thread().ident}] Natural language description completed for {ptype}.")
    except Exception as e:
        print(f"[Thread-{threading.current_thread().ident}] [WARN] Natural language description failed for {ptype}: {e}")

    record = {
        "problem_type": ptype,
        "mode": mode,
        "kernel": kernel,
        "selected_atomic_infos": final_atomics,
        "generation": {
            "problem_description": problem_text_raw,
            "abstract_modeling": modeling_text_raw,
            "gurobi_code": code_text_raw,
            "natural_language_desc": natural_language_desc,
            "timestamp": int(time.time()),
        },
    }
    
    print(f"[Thread-{threading.current_thread().ident}] Completed problem for type: {ptype}")
    return record

def main():
    print(">>> Starting parallel problem generation")
    
    # Configure the progress file path
    progress_path = OUTPUT_PATH.replace('.jsonl', '_progress.json')
    print(f"[INFO] Output file: {OUTPUT_PATH}")
    print(f"[INFO] Progress file: {progress_path}")
    
    # 1. Load reference data
    try:
        pt_stats = load_json(PT_STATS_PATH) if os.path.isfile(PT_STATS_PATH) else {"statistics": []}
        kernels_data = load_json(KERNELS_PATH)
        constraints_catalog = load_json(CONSTRAINTS_CATALOG_PATH)
        atomic_all = load_json(ATOMIC_ALL_PATH)  # Load all atomic information
        
        # Combine sources to identify known Mode A types
        known_types_set = set()
        
        # Source 1: problem_types_atomic_info_statistics.json (full statistics)
        for entry in pt_stats.get("statistics", []):
            known_types_set.add(entry.get("problem_type"))
        
        # Source 2: kernel_combinations.json (kernel combinations)
        kernel_types = set(kernels_data.get("combinations_frequency", {}).keys())
        known_types_set.update(kernel_types)
        
        # Source 3: atomic_info_all.json (atomic information)
        atomic_types = set(set(item.get("problem_type") for item in atomic_all if item.get("problem_type")))
        known_types_set.update(atomic_types)
        
        print(f"\n[Reference source counts]")
        print(f"  - problem_types_atomic_info_statistics.json: {len(pt_stats.get('statistics', []))} types")
        print(f"  - kernel_combinations.json: {len(kernel_types)} types")
        print(f"  - atomic_info_all.json: {len(atomic_types)} types")
        print(f"  - Known types after merging: {len(known_types_set)} types")
            
    except Exception as e:
        print(f"[FATAL] Failed to load reference files: {e}")
        return

    # 2. Build a generation plan for the selected mode
    if USE_MANUAL_MODE:
        # Manual mode: use the requested distribution
        plan = create_manual_plan(MANUAL_TYPE_DISTRIBUTION, PT_STATS_PATH)
    else:
        # Automatic mode: calculate counts from statistics
        try:
            plan = calculate_generation_plan(INPUT_STATS_FILE, TARGET_TOTAL_SAMPLES)
        except Exception as e:
            print(f"[FATAL] Failed to build generation plan: {e}")
            return

    # 3. Initialize the model client
    client = get_client_from_env()
    if not client:
        print("[FATAL] Missing API key; set OPT_ARENA_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")
        return
    model_name = os.getenv("OPT_ARENA_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

    ensure_dir(OUTPUT_DIR)
    
    # 4. Prepare all tasks
    all_tasks = []
    
    for plan_item in plan:
        ptype = plan_item['problem_type']
        count = plan_item['target_samples']
        
        # Choose Mode A or B, including in manual mode
        is_known_type = ptype in known_types_set
        mode = "Mode A (Complex)" if is_known_type else "Mode B (Direct)"
        
        print(f"\nPreparing [{ptype}] -> target: {count} | mode: {mode}")
        
        # Preload the index for Mode A
        idx = {}
        atomic_pool = []
        if is_known_type:
            # Prefer the atomic pool in problem_types_atomic_info_statistics.json
            for entry in pt_stats["statistics"]:
                if entry["problem_type"] == ptype:
                    atomic_pool = entry.get("atomic_infos", [])
                    break
            
            # Otherwise extract atoms from atomic_info_all.json
            if not atomic_pool:
                atomic_pool = list(set(
                    item.get("merged_info") 
                    for item in atomic_all 
                    if item.get("problem_type") == ptype and item.get("merged_info")
                ))
                print(f"  [Info] Loaded {len(atomic_pool)} atomic information items from atomic_info_all.json")
            
            try:
                idx = build_problemtype_atomic_index_cached(ATOMIC_ALL_PATH, ptype)
            except Exception as e:
                print(f"  [Warn] Index build failed for {ptype}: {e}, falling back to Direct Mode.")
                is_known_type = False # Fallback handling
                mode = "Mode B (Direct)"

        # Create a task for each requested problem
        for i in range(count):
            # --- Prepare prompt arguments ---
            mapped_items = []
            
            if is_known_type:
                # Mode A Logic
                kernel = sample_kernel_logic(ptype, kernels_data, atomic_pool)
                # Add randomly sampled atoms for complexity
                current_atomics = list(kernel)
                if len(current_atomics) < 6 and atomic_pool:
                    needed = 6 - len(current_atomics)
                    current_atomics += random.sample(atomic_pool, min(needed, len(atomic_pool)))
                
                mapped_items = map_atomic_logic(current_atomics, idx, constraints_catalog)
                final_atomics = [m["atomic_info"] for m in mapped_items]
            else:
                # Mode B Logic
                kernel = []
                final_atomics = []
                mapped_items = []

            task_args = {
                "ptype": ptype,
                "mapped": mapped_items,
                "kernel": kernel,
                "final_atomics": final_atomics,
                "model": model_name,
                "client": client,
                "mode": mode,
                "task_id": f"{ptype}_{i}"
            }

            all_tasks.append(task_args)
    
    print(f"\nPrepared {len(all_tasks)} generation tasks")
    
    # 5. Load progress and skip completed tasks
    processed_indices = load_progress(progress_path)
    
    # Filter out completed tasks
    tasks_to_process = []
    skipped_count = 0
    for idx, task in enumerate(all_tasks):
        if idx in processed_indices:
            skipped_count += 1
        else:
            task['global_idx'] = idx  # Add a global index for progress tracking
            tasks_to_process.append(task)
    
    print(f"\n[Resuming] Skipped {skipped_count} completed tasks; {len(tasks_to_process)} remain")
    
    if len(tasks_to_process) == 0:
        print("[INFO] All tasks are complete")
        return
    
    # 6. Run generation tasks in parallel
    total_generated = 0
    current_batch_processed_indices = set()  # Indices processed in the current batch
    
    # Create an empty output file if no output or progress exists
    if not os.path.exists(OUTPUT_PATH) and len(processed_indices) == 0:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            pass  # Create an empty file
    
    # Process tasks in batches
    batch_counter = 0
    for batch_start in range(0, len(tasks_to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(tasks_to_process))
        batch_tasks = tasks_to_process[batch_start:batch_end]
        batch_counter += 1
        
        print(f"\nProcessing batch {batch_counter}/{(len(tasks_to_process)-1)//BATCH_SIZE + 1} "
              f"({len(batch_tasks)} tasks)")
        
        batch_records = []
        batch_indices = []  # Record this batch's task indices
        
        # Process the batch in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(generate_single_problem, task): task 
                for task in batch_tasks
            }
            
            # Collect completed results
            for future in as_completed(future_to_task):
                task = future_to_task[future]  # Look up the corresponding task
                record = future.result()
                if record:
                    batch_records.append(record)
                    batch_indices.append(task['global_idx'])
        
        # Write batch results and update progress
        if batch_records:
            write_records_safely(OUTPUT_PATH, batch_records)
            total_generated += len(batch_records)
            
            # Update the set of processed indices
            current_batch_processed_indices.update(batch_indices)
            processed_indices.update(batch_indices)
            
            # Save progress to file
            save_progress(progress_path, processed_indices, total_generated)
            
            print(f"✓ Batch complete; generated {len(batch_records)} records."
                  f"Generated so far: {total_generated} | "
                  f"Progress saved ({len(processed_indices)}/{len(all_tasks)})")
    
    print(f"\n{'='*60}")
    print(f"Complete: generated {total_generated} records")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Final progress: {len(processed_indices)}/{len(all_tasks)} ({len(processed_indices)/len(all_tasks)*100:.1f}%)")
    print(f"{'='*60}")

# ================= Parallel processing helpers =================

def write_records_safely(output_path: str, records: List[Dict[str, Any]]) -> None:
    """
    Write records safely with a thread lock
    """
    lock_file = output_path + ".lock"
    
    # Simple file locking
    while os.path.exists(lock_file):
        time.sleep(0.1)
    
    # Create a lock file
    with open(lock_file, 'w') as f:
        f.write(str(os.getpid()))
    
    try:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
    finally:
        # Remove the lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)

# ================= Progress management helpers =================

def load_progress(progress_path: str) -> set:
    """
    Load processed task indices from the progress file
    The progress file stores only a lightweight integer list
    """
    if not os.path.exists(progress_path):
        return set()
    
    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            processed = set(data.get('processed_indices', []))
            print(f"[INFO] Loaded {len(processed)} completed task indices from progress")
            return processed
    except Exception as e:
        print(f"[WARN] Could not load progress: {e}; starting from the beginning")
        return set()

def save_progress(progress_path: str, processed_indices: set, total_generated: int) -> None:
    """
    Save only processed task indices to the progress file
    """
    progress_data = {
        'processed_indices': sorted(list(processed_indices)),
        'total_generated': total_generated,
        'timestamp': int(time.time())
    }
    
    # Use a temporary file for atomic replacement
    temp_path = progress_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)
    
    # Replace atomically
    if os.path.exists(progress_path):
        os.remove(progress_path)
    os.rename(temp_path, progress_path)

# ================= Prompt builders (general) =================

def get_prompts_complex(problem_type: str, mapped_items: List[Dict]) -> Dict[str, str]:
    """Mode A prompt based on atomic information"""
    
    # 1. Problem Description Prompt
    # lines = [f"Generate a coherent, realistic {problem_type}.", 
    #          "CRITICAL: FULL DATA SPECIFICATION. No placeholders.",
    #          "IMPORTANT: The generated problem MUST explicitly include ALL atomic information listed below.",
    #          "Atomic Constraints to include:"]
    # for i, item in enumerate(mapped_items):
    #     lines.append(f"- {item['atomic_info']}")
    #     for c in item.get('constraints', []):
    #         lines.append(f"  (Type: {c['type']})")
    # p_prompt = "\n".join(lines)

    # 1. Problem Description Prompt

    lines = [
    "You are an expert in operations research.",
    f"Generate ONE realistic and self-contained optimization problem "
    f"of type: {problem_type}.",

    "",
    "### Required Atomic Modeling Information",
    "The problem MUST incorporate ALL the following AMIs "
    "with their intended mathematical meanings:"
    ]

    for i, item in enumerate(mapped_items, 1):
        lines.append(f"{i}. {item['atomic_info']}")
        for c in item.get("constraints", []):
            lines.append(f"   Constraint Type: {c['type']}")

    lines.extend([
    "",
    "### Requirements",

    "1. Problem Structure",
    "- Preserve the essential mathematical structure of the specified problem type.",
    "- Incorporate every required AMI into the problem.",
    "- Ensure that the AMIs are logically compatible.",
    "- Include necessary structural constraints and variable domains.",
    "- Do not introduce unrelated constraints or objectives.",

    "",
    "2. Realistic Scenario",
    "- Construct a coherent real-world optimization scenario.",
    "- Give all decisions, constraints, and objectives clear practical meanings.",
    "- Avoid artificially combining unrelated business requirements.",

    "",
    "3. Complete Numerical Instance",
    "- Provide ALL numerical data needed to solve the problem.",
    "- Specify all relevant sets, dimensions, parameters, and values.",
    "- Provide complete tables or matrices whenever necessary.",
    "- Do not use unspecified parameters or placeholders.",
    "- Ensure consistent units across all data.",

    "",
    "4. Mathematical Validity",
    "- The problem must admit a valid LP or MILP formulation.",
    "- Choose numerical data intended to produce a feasible and bounded instance.",
    "- Avoid nonlinear relationships unless exactly linearizable.",
    "- Keep the instance moderately sized but mathematically non-trivial.",

    "",
    "5. Output",
    "Provide a complete optimization problem with:",
    "- Scenario and background",
    "- Complete numerical data",
    "- Optimization objective",
    "- All requirements and constraints",

    "",
    "Do not provide a mathematical formulation or executable code.",
    "Do not omit any required AMI.",
    "Output ONLY the optimization problem."
    ])

    p_prompt = "\n".join(lines)

    # 2. Modeling Prompt
    m_prompt = """READ a natural language problem description and FORMULATE a clear and correct mathematical optimization model.

STRICT REQUIREMENTS:

1. Output EXACTLY three sections:
   - "### Decision Variables"
   - "### Objective Function"
   - "### Constraints"

2. Core task:
   - Extract decision variables, objective, and constraints from the problem description
   - Do NOT copy sentences directly — convert them into mathematical form
   - Ensure the formulation is logically complete and consistent

3. Style requirements:
   - Keep the output structured and easy to read
   - Use short, natural descriptions where helpful (similar to textbook solutions)
   - Avoid long explanations or introductory paragraphs

4. DO NOT include:
   - LaTeX document structure (\\documentclass, \\begin, \\end, etc.)
   - Display math blocks (NO \\[ ... \\])
   - Explanatory text (e.g., "To solve this problem...", "We formulate...")
   - Redundant or repeated statements

5. Mathematical formatting:
   - Use simple and readable expressions (e.g., π, ≤, ≥, ∑)
   - Keep formulas inline or plain text
   - Avoid heavy LaTeX commands

6. Decision Variables:
   - Format: "- x: description"
   - Clearly define meaning and domain (binary, integer, continuous if applicable)
   - Do NOT include constraints here

7. Objective Function:
   - Start with "Minimize:" or "Maximize:"
   - Expression must match the problem goal exactly

8. Constraints:
   - Use numbered list
   - Each constraint should be concise (1–2 lines max)
   - Add short descriptive labels where helpful (e.g., "Capacity constraint:")
   - Avoid redundant wording

9. CRITICAL (for solver compatibility):
   - The model MUST be linear (LP/MILP)
   - DO NOT include:
     * products of decision variables (x * y)
     * nonlinear functions (e.g., sin, exp, log)
     * division by variables
   - If the problem description implies nonlinearity, rewrite it into a valid linear form if possible

10. Completeness:
   - Include all necessary constraints (e.g., domain constraints, assignment constraints)
   - Do not omit implicit but essential conditions

11. Keep ALL mathematical meaning consistent with the original problem.

---

Example Output:

### Decision Variables
- x_{{i,j}}: 1 if participant i is assigned to car j, 0 otherwise

### Objective Function
Maximize:
∑_{{i}} ∑_{{j}} x_{{i,j}}

### Constraints
1. Each participant is assigned to at most one car:
   ∑_{{j}} x_{{i,j}} ≤ 1 ∀ i

2. Each car is assigned to at most one participant:
   ∑_{{i}} x_{{i,j}} ≤ 1 ∀ j

3. Assignment only allowed if participant is interested:
   x_{{i,j}} = 0 if not allowed

---

Input:
{problem_text}

Output:
"""
    
    # 3. Code Prompt
    c_prompt = """Generate executable Python code using Gurobi to solve the following optimization problem.

## Original Problem Description & Concrete Data
{problem_text}

## Abstract Mathematical Modeling
{modeling_text}

## Critical Modeling Requirement (VERY IMPORTANT)

The final model MUST be a linear or mixed-integer linear program (MILP).

Nonlinear expressions are strictly FORBIDDEN.

DO NOT use:
- Multiplication of decision variables (e.g., x * y)
- Division involving variables (e.g., x / y)
- Nonlinear functions (e.g., max, min, abs, sqrt, exp)
- Conditional logic depending on decision variables (if-else)

---

## If Nonlinear Relationships Exist

You MUST convert them into linear form using standard techniques:

- Binary × Continuous → Big-M linearization
- Absolute value |x| → introduce auxiliary variables
- max/min → auxiliary variables with inequality constraints
- Logical conditions → binary variables + Big-M constraints

---

## Requirements for Code Regeneration

1. **Import Statements**: Must include complete imports:
    ```python
    import gurobipy as gp
    from gurobipy import GRB
    ```

2. **Model Variable**: The Gurobi model must be named `model`.

3. **Constraint Formulation**: 
    - Use `<=` or `>=` instead of strict inequalities `<` or `>`
    - Ensure all constraints from the atomic information are properly implemented
    - For nonlinear relationships, consider linearization or reformulation

4. **Variable Types**: Carefully determine whether each variable should be:
    - Integer (GRB.INTEGER)
    - Continuous (GRB.CONTINUOUS)
    Based on the problem context

5. **Objective Function**: Clearly specify whether it's minimization or maximization based on the problem description.

6. **Output Format**: Print the optimal objective value in a clear format:
    ```python
    print(f"Optimal objective: {{model.objVal}}")
    ```

7. **Code Block**: Provide ONLY ONE code block wrapped in ```python...```

8. **Executable**: The code must be complete and directly executable without modifications.

Please generate the corrected Python code now:
     """
    
    # 4. NL Prompt
    nl_prompt = """You are given a structured description of an optimization problem. 
Rewrite it as a clean natural-language problem statement suitable for an operations research modeling task.

Requirements:
1. Write a coherent and concise problem description in natural language.
2. Preserve ALL numerical values and factual information exactly (demands, costs, capacities, processing times, etc.).
3. Clearly describe the scenario, parameters, constraints, and optimization objective.
4. The final text should read like a standard operations research problem from a textbook or benchmark dataset.
5. Do NOT include modeling instructions such as "decision variables", "sets", "indices", "mathematical equations", or LaTeX formulas.
6. Do NOT include section titles such as "Decision Variables", "Objective Function", or "Constraints".
7. Convert tables or structured data into natural language sentences where appropriate.
8. Do NOT introduce new assumptions or modify the given data.
9. The final description should be concise and focused on the information needed to build an optimization model.

Input:
{problem_text}

Output:
Provide only the rewritten natural-language optimization problem description."""
    
    return {"p": p_prompt, "m": m_prompt, "c": c_prompt, "nl": nl_prompt}

def get_prompts_direct(problem_type: str) -> Dict[str, str]:
    """Mode B prompt for types without metadata"""
    
    # 1. Problem Description
    p_prompt = f"""
    ### Role: Operations Research Expert
    Generate a **HARD and COMPLEX** optimization problem instance for the type: **{problem_type}**.
    
    ### Requirements:
    1. **Context**: Create a realistic industry scenario (e.g., Supply Chain, Scheduling, Energy).
    2. **Complexity**: The problem MUST include at least 2 different types of constraints (e.g., capacity + time windows) and significant data size (e.g., >10 items/nodes).
    3. **Full Data**: You MUST provide ALL specific numbers (costs, demands, capacities) required to solve it. Do NOT use placeholders like "x_i". Hardcode the numbers.
    4. **Structure**: 
       - Narrative
       - Data Tables
       - Objective
       - Constraints description
    """
    
    # 2. Modeling (Same template structure as Mode A)
    m_prompt = """READ a natural language problem description and FORMULATE a clear and correct mathematical optimization model.

STRICT REQUIREMENTS:

1. Output EXACTLY three sections:
   - "### Decision Variables"
   - "### Objective Function"
   - "### Constraints"

2. Core task:
   - Extract decision variables, objective, and constraints from the problem description
   - Do NOT copy sentences directly — convert them into mathematical form
   - Ensure the formulation is logically complete and consistent

3. Style requirements:
   - Keep the output structured and easy to read
   - Use short, natural descriptions where helpful (similar to textbook solutions)
   - Avoid long explanations or introductory paragraphs

4. DO NOT include:
   - LaTeX document structure (\\documentclass, \\begin, \\end, etc.)
   - Display math blocks (NO \\[ ... \\])
   - Explanatory text (e.g., "To solve this problem...", "We formulate...")
   - Redundant or repeated statements

5. Mathematical formatting:
   - Use simple and readable expressions (e.g., π, ≤, ≥, ∑)
   - Keep formulas inline or plain text
   - Avoid heavy LaTeX commands

6. Decision Variables:
   - Format: "- x: description"
   - Clearly define meaning and domain (binary, integer, continuous if applicable)
   - Do NOT include constraints here

7. Objective Function:
   - Start with "Minimize:" or "Maximize:"
   - Expression must match the problem goal exactly

8. Constraints:
   - Use numbered list
   - Each constraint should be concise (1–2 lines max)
   - Add short descriptive labels where helpful (e.g., "Capacity constraint:")
   - Avoid redundant wording

9. CRITICAL (for solver compatibility):
   - The model MUST be linear (LP/MILP)
   - DO NOT include:
     * products of decision variables (x * y)
     * nonlinear functions (e.g., sin, exp, log)
     * division by variables
   - If the problem description implies nonlinearity, rewrite it into a valid linear form if possible

10. Completeness:
   - Include all necessary constraints (e.g., domain constraints, assignment constraints)
   - Do not omit implicit but essential conditions

11. Keep ALL mathematical meaning consistent with the original problem.

---

Example Output:

### Decision Variables
- x_{{i,j}}: 1 if participant i is assigned to car j, 0 otherwise

### Objective Function
Maximize:
∑_{{i}} ∑_{{j}} x_{{i,j}}

### Constraints
1. Each participant is assigned to at most one car:
   ∑_{{j}} x_{{i,j}} ≤ 1 ∀ i

2. Each car is assigned to at most one participant:
   ∑_{{i}} x_{{i,j}} ≤ 1 ∀ j

3. Assignment only allowed if participant is interested:
   x_{{i,j}} = 0 if not allowed

---

Input:
{problem_text}

Output:
"""
    
    # 3. Code (Same template structure as Mode A)
    c_prompt = """Generate executable Python code using Gurobi to solve the following optimization problem.

## Original Problem Description & Concrete Data
{problem_text}

## Abstract Mathematical Modeling
{modeling_text}

## Critical Modeling Requirement (VERY IMPORTANT)

The final model MUST be a linear or mixed-integer linear program (MILP).

Nonlinear expressions are strictly FORBIDDEN.

DO NOT use:
- Multiplication of decision variables (e.g., x * y)
- Division involving variables (e.g., x / y)
- Nonlinear functions (e.g., max, min, abs, sqrt, exp)
- Conditional logic depending on decision variables (if-else)

---

## If Nonlinear Relationships Exist

You MUST convert them into linear form using standard techniques:

- Binary × Continuous → Big-M linearization
- Absolute value |x| → introduce auxiliary variables
- max/min → auxiliary variables with inequality constraints
- Logical conditions → binary variables + Big-M constraints

---

## Requirements for Code Regeneration

1. **Import Statements**: Must include complete imports:
    ```python
    import gurobipy as gp
    from gurobipy import GRB
    ```

2. **Model Variable**: The Gurobi model must be named `model`.

3. **Constraint Formulation**: 
    - Use `<=` or `>=` instead of strict inequalities `<` or `>`
    - Ensure all constraints from the atomic information are properly implemented
    - For nonlinear relationships, consider linearization or reformulation

4. **Variable Types**: Carefully determine whether each variable should be:
    - Integer (GRB.INTEGER)
    - Continuous (GRB.CONTINUOUS)
    Based on the problem context

5. **Objective Function**: Clearly specify whether it's minimization or maximization based on the problem description.

6. **Output Format**: Print the optimal objective value in a clear format:
    ```python
    print(f"Optimal objective: {{model.objVal}}")
    ```

7. **Code Block**: Provide ONLY ONE code block wrapped in ```python...```

8. **Executable**: The code must be complete and directly executable without modifications.

Please generate the corrected Python code now:
     """
    
    # 4. NL (Same as Mode A)
    nl_prompt = """You are given a structured description of an optimization problem. 
Rewrite it as a clean natural-language problem statement suitable for an operations research modeling task.

Requirements:
1. Write a coherent and concise problem description in natural language.
2. Preserve ALL numerical values and factual information exactly (demands, costs, capacities, processing times, etc.).
3. Clearly describe the scenario, parameters, constraints, and optimization objective.
4. The final text should read like a standard operations research problem from a textbook or benchmark dataset.
5. Do NOT include modeling instructions such as "decision variables", "sets", "indices", "mathematical equations", or LaTeX formulas.
6. Do NOT include section titles such as "Decision Variables", "Objective Function", or "Constraints".
7. Convert tables or structured data into natural language sentences where appropriate.
8. Do NOT introduce new assumptions or modify the given data.
9. The final description should be concise and focused on the information needed to build an optimization model.

Input:
{problem_text}

Output:
Provide only the rewritten natural-language optimization problem description."""
    
    return {"p": p_prompt, "m": m_prompt, "c": c_prompt, "nl": nl_prompt}

# ================= Existing kernel-based generation logic (Mode A) =================

def split_kernel_combo(combo_key: str) -> List[str]:
    """Split a kernel combination"""
    parts = []
    tmp = combo_key.split(".,") if ".," in combo_key else [combo_key]
    for t in tmp:
        clean = t.strip().rstrip(".").strip()
        if clean: 
            parts.append(clean)
    return parts

def sample_kernel_logic(problem_type: str, kernels_data: Dict, atomic_pool: List[str]) -> List[str]:
    """Sample a kernel"""
    cf = kernels_data.get("combinations_frequency", {}).get(problem_type)
    if cf:
        items = list(cf.items())
        choices, weights = zip(*items)
        combo_key = random.choices(choices, weights=weights, k=1)[0]
        return split_kernel_combo(combo_key)
    # Fallback
    if atomic_pool:
        return random.sample(atomic_pool, min(3, len(atomic_pool)))
    return []

def build_problemtype_atomic_index_cached(atomic_info_path: str, problem_type: str) -> Dict[str, List[str]]:
    """Build an atomic information index for a problem type"""
    idx = {}
    if ijson is None:
        print("[ERROR] ijson not available, cannot build atomic index")
        return idx
    
    try:
        with open(atomic_info_path, "r", encoding="utf-8") as f:
            parser = ijson.items(f, "item")
            for obj in parser:
                if obj.get("problem_type") == problem_type:
                    mi = normalize_text(obj.get("merged_info", ""))
                    cts = obj.get("constraint_types", [])
                    idx[mi] = list(set(idx.get(mi, []) + cts))
    except Exception as e:
        print(f"[WARN] Failed to build index: {e}")
    
    return idx

def map_atomic_logic(chosen_atomics: List[str], idx: Dict, constraints_catalog: List[Dict]) -> List[Dict]:
    """Map atomic logic to the constraint catalog"""
    cat_map = {c["type"]: c for c in constraints_catalog}
    mapped = []
    for a in chosen_atomics:
        norm_a = normalize_text(a)
        found_types = idx.get(norm_a, [])
        # Fuzzy match
        if not found_types:
            for k, v in idx.items():
                if norm_a in k or k in norm_a:
                    found_types = v
                    break
        
        details = []
        for t in found_types:
            if t in cat_map:
                details.append({
                    "type": t,
                    "description": cat_map[t].get("description"),
                    "index": cat_map[t].get("index")
                })
        mapped.append({"atomic_info": a, "constraints": details})
    return mapped

# ================= Main entry point =================

if __name__ == "__main__":
    main()