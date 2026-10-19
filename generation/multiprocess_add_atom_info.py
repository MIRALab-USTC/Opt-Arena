import json
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os
from collections import defaultdict
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

# File path constants
INPUT_FILE = os.path.join(DATA_DIR, "merged_shuffled_0401_with_atomic_info_with_constraints_classified_fixed.json")
CLASSIFIED_GROUPS_FILE = os.path.join(DATA_DIR, "atomic_info_all.json")
# CLASSIFIED_GROUPS_FILE = os.path.join(ROOT_DIR, "classified_atomic_info_groups.json")
# OUTPUT_FILE = os.path.join(DATA_DIR, "4.0-with_atom_info.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "merged_shuffled_0401_with_merged_atomic_info.json")
UNMATCHED_FILE = os.path.join(DATA_DIR, "unmatched_atomic_info.json")

def load_classified_groups():
    """Load classified atomic information groups"""
    with open(CLASSIFIED_GROUPS_FILE, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    
    # Map each (problem_type, constraint_types_key) pair to its group
    groups_map = {}
    for group in groups:
        problem_type = group.get('problem_type')
        constraint_types = group.get('constraint_types', [])
        
        # Sort constraint types and join them into a key
        constraint_types_key = '|'.join(sorted(constraint_types))
        
        # Use (problem_type, constraint_types_key) as the key
        key = (problem_type, constraint_types_key)
        groups_map[key] = group
    
    return groups_map

def get_problem_type(item):
    """Get the problem type from an item"""
    # Look for the problem type in alternative fields
    return item.get('problem_type', 'Unknown')

def normalize_text(text):
    """Normalize text to reduce mismatches caused by formatting"""
    return ' '.join(str(text).split())

def get_constraint_types(atomic_info, item_constraints=None):
    """Extract constraint types from either supported atomic information format"""
    constraint_types = set()
    
    # Extract types from constraints_with_types
    for constraint in atomic_info.get('constraints_with_types', []):
        constraint_type = constraint.get('type')
        if constraint_type:
            constraint_types.add(constraint_type)

    # If constraints_with_types is absent, match against item['constraints']
    if not constraint_types and item_constraints:
        info_text = normalize_text(atomic_info.get('info', ''))
        related_constraints = {
            normalize_text(c) for c in atomic_info.get('related_constraints', []) if c
        }

        for constraint in item_constraints:
            constraint_type = constraint.get('type')
            if not constraint_type:
                continue

            source_info = normalize_text(constraint.get('source_info', ''))
            constraint_text = normalize_text(constraint.get('constraint', ''))

            # First match on both source_info and constraint text
            if info_text and source_info == info_text:
                if not related_constraints or constraint_text in related_constraints:
                    constraint_types.add(constraint_type)
                    continue

            # Fallback: match on constraint text alone
            if related_constraints and constraint_text in related_constraints:
                constraint_types.add(constraint_type)
    
    # Return a sorted list
    return sorted(list(constraint_types))

def process_item(args):
    """Process one problem item in a worker process"""
    item, item_idx, groups_map = args
    
    problem_type = get_problem_type(item)
    unmatched_entries = []
    updated_atomic_info = []
    
    # Process each atomic information item
    for atomic_idx, atomic_info in enumerate(item.get('atomic_information', [])):
        # Get constraint types
        constraint_types = get_constraint_types(atomic_info, item.get('constraints', []))
        constraint_types_key = '|'.join(constraint_types)
        
        # Find a matching group
        key = (problem_type, constraint_types_key)
        matched_group = groups_map.get(key)
        
        # Create a new atomic information object without modifying the input
        new_atomic_info = atomic_info.copy()
        
        if matched_group:
            # Add the atom_info field
            new_atomic_info['atom_info'] = matched_group.get('merged_info', 'No matching generic description found')
        else:
            # Record unmatched entries
            unmatched_entry = {
                'problem_index': item_idx,
                'atomic_index': atomic_idx,
                'problem_type': problem_type,
                'constraint_types': constraint_types,
                'atomic_info': atomic_info.get('info', ''),
                'related_constraints': atomic_info.get('related_constraints', [])
            }
            unmatched_entries.append(unmatched_entry)
            
            # Add the default atom_info field
            new_atomic_info['atom_info'] = 'No matching generic description found'
        
        updated_atomic_info.append(new_atomic_info)
    
    # Update atomic information in the item
    item['atomic_information'] = updated_atomic_info
    
    return item, unmatched_entries

def process_data_parallel(data, groups_map, num_processes=None):
    """Process data in parallel and add atom_info fields"""
    if num_processes is None:
        num_processes = min(cpu_count(), len(data))
    
    # Prepare arguments
    args = [(item, idx, groups_map) for idx, item in enumerate(data)]
    
    unmatched_entries = []
    processed_data = []
    
    print(f"Starting {num_processes} processes for parallel processing...")
    
    # Process data with a multiprocessing pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_item, args)
    
    # Combine results
    for item, unmatched in results:
        processed_data.append(item)
        unmatched_entries.extend(unmatched)
    
    return processed_data, unmatched_entries

def main():
    print("Loading classified atomic info groups...")
    groups_map = load_classified_groups()
    print(f"Loaded {len(groups_map)} classified groups")
    
    print("Loading problem data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")
    
    print("Processing data and adding atom_info fields...")
    processed_data, unmatched_entries = process_data_parallel(data, groups_map)
    
    print("Saving processed data...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print("Saving unmatched entries...")
    with open(UNMATCHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(unmatched_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed!")
    print(f"  - Processed {len(processed_data)} problems")
    print(f"  - Found {len(unmatched_entries)} unmatched atomic info entries")
    print(f"  - Results saved to {OUTPUT_FILE}")
    print(f"  - Unmatched entries saved to {UNMATCHED_FILE}")

if __name__ == "__main__":
    main()