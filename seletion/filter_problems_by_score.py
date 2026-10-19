import json
import os
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

def load_data(problem_file, atomic_info_count_file):
    """Load problems and atomic information occurrence counts."""
    with open(problem_file, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    with open(atomic_info_count_file, 'r', encoding='utf-8') as f:
        atomic_info_counts = json.load(f)
    
    return problems, atomic_info_counts

def calculate_overlap(problem, atomic_info_counts):
    """Calculate a problem's mean atomic information occurrence count."""
    atomic_infos = problem.get('atomic_information', [])
    if not atomic_infos:
        return 0
    
    total_count = 0
    for atom_info in atomic_infos:
        info_text = atom_info.get('atom_info', '').strip()
        if info_text and info_text in atomic_info_counts:
            total_count += atomic_info_counts[info_text]['count']
    
    # Average over all atomic information entries, including unmatched entries.
    return total_count / len(atomic_infos)

def calculate_difficulty(problem, alpha=0.1):
    """Calculate difficulty with a constraint count adjustment."""
    base_difficulty = problem.get('difficulty', 0)
    
    # Find the largest related constraint count among the atomic information entries.
    max_constraints = 0
    atomic_infos = problem.get('atomic_information', [])
    for atom_info in atomic_infos:
        constraints = atom_info.get('related_constraints', [])
        max_constraints = max(max_constraints, len(constraints))
    
    # Difficulty = base difficulty + alpha * maximum constraint count.
    return base_difficulty + alpha * max_constraints

def calculate_score(problem, atomic_info_counts, lambda_val=1.0, alpha=0.1):
    """Calculate the difficulty score minus the overlap score."""
    difficulty_value = calculate_difficulty(problem, alpha)
    overlap_value = calculate_overlap(problem, atomic_info_counts)
    score = lambda_val * difficulty_value - overlap_value
    return score

def calculate_score_variant_a(problem, atomic_info_counts, lambda_val=0.0, alpha=0.1):
    """
    Variant A: penalize overlap only (score = -overlap).
    Use with the coverage-preserving filter.
    """
    # Difficulty is deliberately excluded from this variant.
    overlap_value = calculate_overlap(problem, atomic_info_counts)
    
    # More redundant problems receive lower scores and are considered first.
    score = - overlap_value
    
    return score

def calculate_score_variant_b(problem, atomic_info_counts, lambda_val=1.0, alpha=0.1):
    """
    Variant B: score by difficulty only, without an overlap penalty.
    Use with the coverage-preserving filter.
    """
    difficulty_value = calculate_difficulty(problem, alpha)
    
    # Ignore overlap in this variant.
    score = lambda_val * difficulty_value
    
    return score

def filter_problems_variant_c(problems, atomic_info_counts, target_count=None, lambda_val=1.0, alpha=0.1):
    """
    Variant C: remove the lowest scoring problems without coverage protection.
    """
    # Calculate scores.
    problem_scores = []
    for problem in problems:
        score = calculate_score(problem, atomic_info_counts, lambda_val, alpha)
        problem_scores.append((problem['idx'], score, problem))
    
    # Sort from lowest to highest score.
    problem_scores.sort(key=lambda x: x[1])
    
    print(f"Initial problem count: {len(problems)}")
    
    # Remove the lowest scoring problems directly.
    delete_count = len(problems) - target_count if target_count else 0
    
    if delete_count > 0:
        # Delete the lowest scoring problems.
        deleted_problems = [p for _, _, p in problem_scores[:delete_count]]
        # Keep the rest.
        kept_problems = [p for _, _, p in problem_scores[delete_count:]]
    else:
        deleted_problems = []
        kept_problems = [p for _, _, p in problem_scores]
        
    print(f"Deleted problem count: {len(deleted_problems)}")
    print(f"Kept problem count: {len(kept_problems)}")
    
    return kept_problems, deleted_problems

def count_scenario_occurrences(problems):
    """Count occurrences of each nonempty scenario."""
    scenario_counts = defaultdict(int)
    for problem in problems:
        scenario = problem.get('scenario', '')
        if scenario:
            scenario_counts[scenario] += 1
    return scenario_counts

def filter_problems(problems, atomic_info_counts, target_count=None, lambda_val=1.0, alpha=0.1):
    """
    Select problems by score while preserving atomic information and scenario coverage.
    
    Args:
        problems: Problem records.
        atomic_info_counts: Atomic information occurrence statistics.
        target_count: Desired number of retained problems, or None.
        lambda_val: Difficulty weight.
        alpha: Constraint count weight.
    
    Returns:
        kept_problems: Retained problem records.
        deleted_problems: Removed problem records.
    """
    # Calculate each problem's score.
    problem_scores = []
    for problem in problems:
        score = calculate_score(problem, atomic_info_counts, lambda_val, alpha)
        problem_scores.append((problem['idx'], score, problem))
    
    # Sort from lowest to highest score.
    problem_scores.sort(key=lambda x: x[1])
    
    # Count initial atomic information and scenario occurrences.
    atomic_info_occurrences = defaultdict(int)
    scenario_occurrences = defaultdict(int)
    
    for problem in problems:
        # Count atomic information.
        for atom_info in problem.get('atomic_information', []):
            info_text = atom_info.get('atom_info', '').strip()
            if info_text:
                atomic_info_occurrences[info_text] += 1
        
        # Count scenarios.
        scenario = problem.get('scenario', '')
        if scenario:
            scenario_occurrences[scenario] += 1
    
    print(f"Initial atomic information types: {len(atomic_info_occurrences)}")
    print(f"Initial scenario types: {len(scenario_occurrences)}")
    print(f"Initial problem count: {len(problems)}")
    
    # Initialize result lists.
    kept_problems = []
    deleted_problems = []
    
    # Attempt removal starting with the lowest score.
    for idx, score, problem in problem_scores:
        can_delete = True
        
        # Preserve at least one occurrence of each atomic information type.
        temp_atomic_info_counts = atomic_info_occurrences.copy()
        for atom_info in problem.get('atomic_information', []):
            info_text = atom_info.get('atom_info', '').strip()
            if info_text:
                temp_atomic_info_counts[info_text] -= 1
                # Keep the problem if this is the last occurrence.
                if temp_atomic_info_counts[info_text] == 0:
                    can_delete = False
                    break
        
        # Preserve at least one occurrence of each scenario.
        if can_delete:
            scenario = problem.get('scenario', '')
            if scenario:
                temp_scenario_count = scenario_occurrences[scenario] - 1
                # Keep the problem if this is the last occurrence.
                if temp_scenario_count == 0:
                    can_delete = False
        
        # Update occurrence counts after removal.
        if can_delete:
            # Update atomic information counts.
            for atom_info in problem.get('atomic_information', []):
                info_text = atom_info.get('atom_info', '').strip()
                if info_text:
                    atomic_info_occurrences[info_text] -= 1
            
            # Update scenario counts.
            scenario = problem.get('scenario', '')
            if scenario:
                scenario_occurrences[scenario] -= 1
            
            deleted_problems.append(problem)
        else:
            kept_problems.append(problem)
        
        # Stop removing problems once the target count is reached.
        if target_count and (len(problems) - len(deleted_problems)) <= target_count:
            # Keep all unprocessed problems.
            remaining_problems = [p for _, _, p in problem_scores[problem_scores.index((idx, score, problem))+1:]]
            kept_problems.extend(remaining_problems)
            break
    
    print(f"Deleted problem count: {len(deleted_problems)}")
    print(f"Kept problem count: {len(kept_problems)}")
    
    # Count remaining atomic information and scenario types.
    final_atomic_info_count = sum(1 for count in atomic_info_occurrences.values() if count > 0)
    final_scenario_count = sum(1 for count in scenario_occurrences.values() if count > 0)
    
    print(f"Remaining atomic information types: {final_atomic_info_count}")
    print(f"Remaining scenario types: {final_scenario_count}")
    
    return kept_problems, deleted_problems

def main():
    # Input and output files are stored in the configurable local data directory.
    problem_file = os.path.join(DATA_DIR, "6-with_difficulties_valid_scenarios_only.json")
    atomic_info_count_file = os.path.join(DATA_DIR, "atomic_info_occurrences.json")
    output_kept_file = os.path.join(DATA_DIR, "filtered_kept_problems_200.json")
    output_deleted_file = os.path.join(DATA_DIR, "filtered_deleted_problems_200.json")
    
    # Set to None to remove the target count limit.
    target_count = 5000
    lambda_val = 1.0
    alpha = 0.1
    
    # Load input data.
    problems, atomic_info_counts = load_data(problem_file, atomic_info_count_file)
    
    # Filter problems.
    kept_problems, deleted_problems = filter_problems(
        problems, atomic_info_counts, target_count, lambda_val, alpha
    )
    
    # Save results.
    with open(output_kept_file, 'w', encoding='utf-8') as f:
        json.dump(kept_problems, f, ensure_ascii=False, indent=2)
    
    with open(output_deleted_file, 'w', encoding='utf-8') as f:
        json.dump(deleted_problems, f, ensure_ascii=False, indent=2)
    
    print(f"Kept problems saved to: {output_kept_file}")
    print(f"Deleted problems saved to: {output_deleted_file}")

if __name__ == "__main__":
    main()
