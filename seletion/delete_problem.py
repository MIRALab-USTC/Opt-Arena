import os
import json
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

def remove_missing_atomic_info_entries():
    """
    Remove entries indexed by missing_atomic_info_indices.json from 1-info_and_constraints.json
    """
    # Read indices with missing atomic information
    with open(os.path.join(DATA_DIR, "missing_atomic_info_indices.json"), 'r') as f:
        missing_indices = set(json.load(f))
    
    # Read input data
    with open(os.path.join(DATA_DIR, "1-info_and_constraints.json"), 'r') as f:
        original_data = json.load(f)
    
    # Filter out entries with missing atomic information
    filtered_data = []
    for index, item in enumerate(original_data):
        if index not in missing_indices:
            filtered_data.append(item)
    
    # Save filtered data
    with open(os.path.join(DATA_DIR, "1-info_and_constraints_filtered.json"), 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Original entries: {len(original_data)}")
    print(f"Filtered entries: {len(filtered_data)}")
    print(f"Removed entries: {len(missing_indices)}")

# Run filtering
if __name__ == "__main__":
    remove_missing_atomic_info_entries()