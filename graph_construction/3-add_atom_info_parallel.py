import json
import multiprocessing as mp
from collections import defaultdict
from functools import partial
import os
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

# File path constants
INPUT_FILE = os.path.join(DATA_DIR, "3-with_constraint_types_and_problem_types.json")
CLASSIFIED_GROUPS_FILE = os.path.join(ROOT_DIR, "classified_atomic_info_groups.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "4.0-with_atom_info.json")
UNMATCHED_FILE = os.path.join(DATA_DIR, "unmatched_atomic_info.json")

# Batch parameters
BATCH_SIZE = 100  # Items per batch
WRITE_INTERVAL = 10  # Number of batches between writes

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
        groups_map[key] = group.get('merged_info', 'No matching generic description found')
    
    return groups_map

def get_problem_type(item):
    """Get the problem type from an item"""
    return item.get('problem_type', 'Unknown')

def get_constraint_types(atomic_info):
    """Extract constraint types from atomic information"""
    constraint_types = set()
    
    # Extract types from constraints_with_types
    for constraint in atomic_info.get('constraints_with_types', []):
        constraint_type = constraint.get('type')
        if constraint_type:
            constraint_types.add(constraint_type)
    
    # Return a sorted list
    return sorted(list(constraint_types))

def process_single_item(args):
    """Process one problem item"""
    item_idx, item, groups_map = args
    problem_type = get_problem_type(item)
    unmatched_entries = []
    updated_atomic_info = []
    
    # Process each atomic information item
    for atomic_idx, atomic_info in enumerate(item.get('atomic_information', [])):
        # Get constraint types
        constraint_types = get_constraint_types(atomic_info)
        constraint_types_key = '|'.join(constraint_types)
        
        # Find a matching group
        key = (problem_type, constraint_types_key)
        merged_info = groups_map.get(key, 'No matching generic description found')
        
        # Update atomic information
        atomic_info_copy = atomic_info.copy()
        atomic_info_copy['atom_info'] = merged_info
        updated_atomic_info.append(atomic_info_copy)
        
        # Record entries without a match
        if merged_info == 'No matching generic description found':
            unmatched_entry = {
                'problem_index': item_idx,
                'atomic_index': atomic_idx,
                'problem_type': problem_type,
                'constraint_types': constraint_types,
                'atomic_info': atomic_info.get('info', ''),
                'related_constraints': atomic_info.get('related_constraints', [])
            }
            unmatched_entries.append(unmatched_entry)
    
    # Update atomic information in the item
    item_copy = item.copy()
    item_copy['atomic_information'] = updated_atomic_info
    
    return item_copy, unmatched_entries

def process_data_parallel(data, groups_map, num_processes=None):
    """Process data in parallel and add atom_info fields"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Total item and batch counts
    total_items = len(data)
    total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Processing {total_items} items in {total_batches} batches of {BATCH_SIZE}")
    
    # Initialize result and unmatched-entry lists
    all_processed_data = []
    all_unmatched_entries = []
    
    # Process in batches
    batch_count = 0
    processed_items = 0
    
    for i in range(0, total_items, BATCH_SIZE):
        batch_start = i
        batch_end = min(i + BATCH_SIZE, total_items)
        batch_data = data[batch_start:batch_end]
        
        # Prepare arguments
        process_args = [(item_idx, item, groups_map) for item_idx, item in enumerate(batch_data, batch_start)]
        
        # Process in parallel with a process pool
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_item, process_args)
        
        # Collect results
        batch_processed_data = []
        batch_unmatched_entries = []
        
        for item_result, unmatched_entries in results:
            batch_processed_data.append(item_result)
            batch_unmatched_entries.extend(unmatched_entries)
        
        # Add to the overall results
        all_processed_data.extend(batch_processed_data)
        all_unmatched_entries.extend(batch_unmatched_entries)
        
        # Update the processed count
        batch_count += 1
        processed_items += len(batch_data)
        
        # Display progress
        progress = processed_items / total_items * 100
        print(f"Progress: {processed_items}/{total_items} ({progress:.1f}%) - batch {batch_count}/{total_batches} completed")
        
        # Write every WRITE_INTERVAL batches
        if batch_count % WRITE_INTERVAL == 0 or batch_count == total_batches:
            print(f"Writing results after batch {batch_count}...")
            # Write current results
            temp_output_file = OUTPUT_FILE.replace('.json', f'_temp_{batch_count}.json')
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_processed_data, f, indent=2, ensure_ascii=False)
            print(f"Temporary results saved to {temp_output_file}")
    
    return all_processed_data, all_unmatched_entries

def main():
    print("Loading classified atomic info groups...")
    groups_map = load_classified_groups()
    print(f"Loaded {len(groups_map)} classified groups")
    
    print("Loading problem data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")
    
    print("Processing data and adding atom_info fields in parallel...")
    start_time = time.time()
    processed_data, unmatched_entries = process_data_parallel(data, groups_map)
    end_time = time.time()
    
    print("Saving final processed data...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print("Saving unmatched entries...")
    with open(UNMATCHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(unmatched_entries, f, indent=2, ensure_ascii=False)
    
    # Remove temporary files
    for i in range(WRITE_INTERVAL, (len(data) + BATCH_SIZE - 1) // BATCH_SIZE + 1, WRITE_INTERVAL):
        temp_file = OUTPUT_FILE.replace('.json', f'_temp_{i}.json')
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"Processing completed!")
    print(f"  - Processed {len(processed_data)} problems")
    print(f"  - Found {len(unmatched_entries)} unmatched atomic info entries")
    print(f"  - Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"  - Results saved to {OUTPUT_FILE}")
    print(f"  - Unmatched entries saved to {UNMATCHED_FILE}")

if __name__ == "__main__":
    main()