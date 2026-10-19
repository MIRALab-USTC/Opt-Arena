import os
import json
from typing import List, Dict, Any
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

def load_constraints(file_path: str) -> List[Dict]:
    """Load constraint type data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_constraints(constraints: List[Dict], file_path: str):
    """Save constraint type data"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(constraints, f, indent=2, ensure_ascii=False)

def calculate_difficulty_score(constraint: Dict) -> Dict:
    """
    Calculate a constraint type's difficulty score
    
    Mapping rules:
    - auxiliary_variables: "Yes" -> 1, "No" -> 0
    - advanced_techniques: "Yes" -> 1, "No" -> 0
    - variable_types_count: 1->0, 2->0.33, 3->0.67, 4->1
    
    Return a dictionary with three components and the total score
    """
    difficulty = constraint.get('difficulty', {})
    
    # Score auxiliary variables
    aux_var = difficulty.get('auxiliary_variables', 'No')
    aux_score = 1 if aux_var.lower() == 'yes' else 0
    
    # Score advanced techniques
    adv_tech = difficulty.get('advanced_techniques', 'No')
    adv_score = 1 if adv_tech.lower() == 'yes' else 0
    
    # Score the number of variable types
    var_count = difficulty.get('variable_types_count', 1)
    var_score_map = {1: 0, 2: 0.33, 3: 0.67, 4: 1}
    var_score = var_score_map.get(var_count, 0)
    
    # Calculate the weighted total
    total_score = 0.25*aux_score + 0.35*adv_score + 0.15*var_score
    
    return {
        'auxiliary_variables_score': aux_score,
        'advanced_techniques_score': adv_score,
        'variable_types_score': var_score,
        'difficulty_score': total_score
    }

def main():
    # File paths
    input_file = os.path.join(DATA_DIR, "constraints_with_difficulty-train.json")
    output_file = os.path.join(ROOT_DIR, "constraints_with_difficulty_scores-train.json")
    
    # Load constraint data
    print("Loading constraints data...")
    constraints = load_constraints(input_file)
    print(f"Loaded {len(constraints)} constraint types")
    
    # Calculate the difficulty score for each constraint
    for i, constraint in enumerate(constraints):
        print(f"Processing constraint {i+1}/{len(constraints)}: {constraint['type']}")
        
        # Check whether a difficulty assessment exists
        if 'difficulty' not in constraint:
            print(f"  Warning: No difficulty assessment found for {constraint['type']}")
            continue
            
        # Calculate the difficulty score
        scores = calculate_difficulty_score(constraint)
        
        # Add scores to the constraint data
        constraint['difficulty_scores'] = scores
        
        print(f"  Scores - Aux: {scores['auxiliary_variables_score']}, "
              f"Adv: {scores['advanced_techniques_score']}, "
              f"Var: {scores['variable_types_score']}, "
              f"Total: {scores['difficulty_score']}")
    
    # Save results
    save_constraints(constraints, output_file)
    print(f"All difficulty scores calculated and saved to {output_file}")
    
    # Print summary statistics
    print("\nDifficulty Score Statistics:")
    scores = [c.get('difficulty_scores', {}).get('difficulty_score', 0) for c in constraints]
    if scores:
        print(f"  Min Score: {min(scores)}")
        print(f"  Max Score: {max(scores)}")
        print(f"  Average Score: {sum(scores)/len(scores):.2f}")

if __name__ == "__main__":
    main()