import json
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple
import os
import math
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

def load_constraint_difficulties(file_path: str) -> Dict[str, float]:
    """Load constraint type difficulty data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        constraints = json.load(f)
    
    # Map constraint types to difficulty scores
    constraint_difficulties = {}
    for constraint in constraints:
        constraint_type = constraint['type']
        # Use difficulty_scores.difficulty_score as the constraint difficulty
        difficulty_score = constraint['difficulty_scores']['difficulty_score']
        constraint_difficulties[constraint_type] = difficulty_score
    
    return constraint_difficulties

def load_problem_data(file_path: str) -> List[Dict]:
    """Load problem data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_problem_data(data: List[Dict], file_path: str):
    """Save problem data"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def calculate_constraint_difficulty(constraint: Dict, constraint_difficulties: Dict[str, float]) -> float:
    """
    Calculate the difficulty of one constraint
    Constraint difficulty = type difficulty + 0.25 * dimensionality difficulty
    """
    # Look up the constraint type difficulty
    constraint_type = constraint.get('type', '')
    base_difficulty = constraint_difficulties.get(constraint_type, 0.0)
    
    # Calculate dimensionality difficulty
    dimensionality = constraint.get('dimensionality', 'one-dimensional')
    dimension_difficulty = 1.0 if dimensionality == 'multi-dimensional' else 0.0
    
    # Total difficulty = type difficulty + 0.25 * dimensionality difficulty
    total_difficulty = base_difficulty + 0.25 * dimension_difficulty
    
    return total_difficulty

def calculate_atomic_info_difficulty(atomic_info: Dict, problem_constraints: List[Dict], constraint_difficulties: Dict[str, float]) -> float:
    """
    Atomic information difficulty is the sum of its related constraint difficulties
    """
    total_difficulty = 0.0
    
    # Get constraints related to the atomic information
    related_constraints_text = atomic_info.get('related_constraints', [])
    
    # Find related constraints among all problem constraints
    for constraint in problem_constraints:
        constraint_text = constraint.get('constraint', '')
        # Check whether the constraint relates to the atomic information
        if constraint_text in related_constraints_text:
            # Calculate and add this constraint's difficulty
            constraint_difficulty = calculate_constraint_difficulty(constraint, constraint_difficulties)
            total_difficulty += constraint_difficulty
    
    # In OptMATH data, constraints_with_types contains constraints and their types
    constraints_with_types = atomic_info.get('constraints_with_types', [])
    for constraint in constraints_with_types:
        constraint_difficulty = calculate_constraint_difficulty(constraint, constraint_difficulties)
        total_difficulty += constraint_difficulty
    
    return total_difficulty

def calculate_problem_difficulty_single(args) -> Tuple[int, Dict]:
    """
    Calculate one problem's difficulty for parallel processing
    Return the problem index and processed problem data
    """
    problem, problem_index, constraint_difficulties = args
    
    # Deep-copy the problem to avoid modifying the input
    problem_copy = json.loads(json.dumps(problem))
    
    atomic_infos = problem_copy.get('atomic_information', [])
    constraints = problem_copy.get('constraints', []) if 'constraints' in problem_copy else []
    
    if not atomic_infos:
        problem_copy['difficulty'] = 0.0
        return problem_index, problem_copy
    
    # Calculate each atomic information item's difficulty
    atomic_difficulties = []
    for atomic_info in atomic_infos:
        difficulty = calculate_atomic_info_difficulty(atomic_info, constraints, constraint_difficulties)
        atomic_difficulties.append(difficulty)
        # Attach difficulty to the atomic information
        atomic_info['difficulty'] = difficulty
    
    # Problem difficulty = highest atomic information difficulty
    problem_difficulty = max(atomic_difficulties) if atomic_difficulties else 0.0
    problem_copy['difficulty'] = problem_difficulty
    
    # Calculate and attach difficulty to each constraint, if present
    for constraint in constraints:
        constraint_difficulty = calculate_constraint_difficulty(constraint, constraint_difficulties)
        constraint['difficulty'] = constraint_difficulty
    
    return problem_index, problem_copy

def process_problems_parallel(problems: List[Dict], constraint_difficulties: Dict[str, float], num_processes: int = None) -> List[Dict]:
    """Process problems in parallel and calculate difficulty"""
    if num_processes is None:
        num_processes = min(cpu_count(), len(problems))
    
    # Prepare arguments
    args = [(problem, idx, constraint_difficulties) for idx, problem in enumerate(problems)]
    
    print(f"Starting {num_processes} processes for parallel processing...")
    
    # Process data with a multiprocessing pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_problem_difficulty_single, args)
    
    # Restore the original result order
    sorted_results = sorted(results, key=lambda x: x[0])
    processed_problems = [problem for _, problem in sorted_results]
    
    return processed_problems

def process_problems_in_batches(problems: List[Dict], constraint_difficulties: Dict[str, float], 
                               batch_size: int = 10, num_processes: int = None, 
                               write_interval: int = 1, output_file: str = "") -> List[Dict]:
    """
    Process problems in batches and periodically write results
    
    Args:
        problems: list of problems
        constraint_difficulties: constraint difficulty mapping
        batch_size: number of problems per batch
        num_processes: number of parallel processes
        write_interval: number of batches between writes
        output_file: output file path
    """
    total_problems = len(problems)
    total_batches = math.ceil(total_problems / batch_size)
    
    # Load previous progress if the output file exists
    if os.path.exists(output_file):
        print(f"Loading existing progress file: {output_file}")
        processed_problems = load_problem_data(output_file)
        start_batch = len(processed_problems) // batch_size
        print(f"Resuming from batch {start_batch + 1}")
    else:
        processed_problems = []
        start_batch = 0
    
    print(f"{total_batches} batches total; {batch_size} problems per batch; write every {write_interval} batches")
    
    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_problems)
        batch_problems = problems[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} (problems {start_idx}-{end_idx-1})")
        
        # Process this batch in parallel
        batch_results = process_problems_parallel(batch_problems, constraint_difficulties, num_processes)
        processed_problems.extend(batch_results)
        
        # Save results at the configured interval
        if (batch_idx + 1) % write_interval == 0 or batch_idx == total_batches - 1:
            print(f"Writing results for the first {len(processed_problems)} problems...")
            save_problem_data(processed_problems, output_file)
            print(f"Saved progress to {output_file}")
    
    return processed_problems

def main():
    # Default parameters
    num_processes = 16  # Default number of parallel processes
    batch_size = 100    # Default batch size
    write_interval = 10 # Write every 10 batches by default
    
    # File paths
    constraint_difficulty_file = os.path.join(ROOT_DIR, "constraints_with_difficulty_scores-train.json")
    # problem_data_file = os.path.join(DATA_DIR, "5-with_scenarios_regenerated.json")
    # output_file = os.path.join(DATA_DIR, "6-with_difficulties.json")
    problem_data_file = os.path.join(DATA_DIR, "merged_shuffled_0401_with_merged_atomic_info_cleaned-2_fixed_with_dim.json")
    output_file = os.path.join(DATA_DIR, "merged_shuffled_0401_with_difficulties.json")

    # Load constraint difficulty data
    print("Loading constraint difficulties...")
    constraint_difficulties = load_constraint_difficulties(constraint_difficulty_file)
    print(f"Loaded difficulties for {len(constraint_difficulties)} constraint types")
    
    # Load problem data
    print("Loading problem data...")
    problems = load_problem_data(problem_data_file)
    print(f"Loaded {len(problems)} problems")
    
    # Calculate problem difficulty in batches
    print(f"Calculating problem difficulties with {num_processes} processes, batch size {batch_size}, write interval {write_interval}...")
    processed_problems = process_problems_in_batches(
        problems, 
        constraint_difficulties, 
        batch_size=batch_size, 
        num_processes=num_processes, 
        write_interval=write_interval, 
        output_file=output_file
    )
    
    print(f"Processing completed. Total {len(processed_problems)} problems processed.")

if __name__ == "__main__":
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()