import json
import os
from typing import List, Dict, Any
import re
import openai
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock
import copy
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

# Initialize OpenAI client
API_KEY = os.getenv("OPT_ARENA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("OPT_ARENA_BASE_URL", "https://yunwu.ai/v1")
LLM_MODEL = os.getenv("OPT_ARENA_MODEL", "gpt-4o-mini")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None

# Thread lock for file writing operations
file_lock = Lock()
constraint_pool_lock = Lock()

def load_existing_constraints(constraints_file: str) -> List[Dict[str, Any]]:
    """
    Load existing constraint types from constraints.json
    """
    with open(constraints_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """
    Load the dataset with constraints to be classified
    """
    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(dataset: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save the classified dataset
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

def save_dataset_batch(dataset: List[Dict[str, Any]], output_file: str, indices: List[int]) -> None:
    """
    Save a batch of processed items to the dataset
    """
    with file_lock:
        # Load existing data
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = [None] * len(dataset)
        
        # Update processed items
        for idx in indices:
            existing_data[idx] = dataset[idx]
        
        # Save updated data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

def save_constraint_pool(constraint_pool: List[Dict[str, Any]], constraints_file: str) -> None:
    """
    Save the updated constraint pool
    """
    with file_lock:
        with open(constraints_file, 'w', encoding='utf-8') as f:
            json.dump(constraint_pool, f, indent=2, ensure_ascii=False)

def load_progress(progress_file: str) -> set:
    """
    Load progress from file to support resuming
    """
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_items: set, progress_file: str) -> None:
    """
    Save progress to file
    """
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_items), f, indent=2, ensure_ascii=False)

def get_constraint_pool_snapshot(constraint_pool: List[Dict[str, Any]], lock) -> List[Dict[str, Any]]:
    """
    Get a thread-safe snapshot of the constraint pool
    """
    with lock:
        return copy.deepcopy(constraint_pool)

def format_constraint_pool_for_prompt(constraint_pool: List[Dict[str, Any]]) -> str:
    """
    Format the constraint pool for inclusion in the LLM prompt
    """
    formatted = ""
    for i, constraint_type in enumerate(constraint_pool):
        formatted += f"{i+1}. Type: {constraint_type['type']}\n"
        formatted += f"   Description: {constraint_type['description']}\n"
        formatted += f"   Example: {constraint_type['example']}\n\n"
    return formatted

def log_new_constraint_type(problem_index: int, constraint_index: int, constraint_text: str, 
                           new_type: str, description: str, example: str, log_file: str = "new_constraint_types_log.json") -> None:
    """
    Log information about newly created constraint types
    """
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "problem_index": problem_index,
        "constraint_index": constraint_index,
        "constraint_text": constraint_text,
        "new_type": new_type,
        "description": description,
        "example": example
    }
    
    # Thread-safe log writing
    with file_lock:
        # Load existing log or create new one
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Append new entry
        log_data.append(log_entry)
        
        # Save updated log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

def create_classification_prompt(constraint_text: str, source_info: str, constraint_pool: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for the LLM to classify a constraint
    """
    prompt = f"""You are an expert in operations research and mathematical optimization. Your task is to classify mathematical constraints into predefined types.

CONSTRAINT TO CLASSIFY:
Constraint Expression: {constraint_text}
Source Information: {source_info}

AVAILABLE CONSTRAINT TYPES:
{format_constraint_pool_for_prompt(constraint_pool)}

IMPORTANT INSTRUCTIONS:
1. You MUST select a type from the AVAILABLE CONSTRAINT TYPES list above.
2. Only create a NEW constraint type if the constraint ABSOLUTELY cannot be classified using any existing type.
3. Constraint type additions should be MINIMAL and ONLY when genuinely necessary.
4. When selecting a type, consider both the mathematical expression and the source information.
5. Pay close attention to the descriptions and examples of each constraint type.
6. Determine if the constraint is one-dimensional or multi-dimensional:
   - One-dimensional: Contains only single subscript variables (e.g., x, y, x_i, y_j)
   - Multi-dimensional: Contains multiple subscript variables (e.g., x_ij, y_ijk)

Response Format:
If using existing type:
{{
  "type": "existing_type_name",
  "dimensionality": "one-dimensional or multi-dimensional"
}}

If creating new type (ONLY when absolutely necessary):
{{
  "type": "new_type_name",
  "description": "detailed_description_of_when_this_type_applies",
  "example": "mathematical_example_of_this_type",
  "dimensionality": "one-dimensional or multi-dimensional"
}}

Respond ONLY with a valid JSON object in the format above. No other text."""

    return prompt

def create_new_type_prompt(constraint_text: str, source_info: str) -> str:
    """
    Create a prompt for the LLM to generate a new constraint type
    """
    prompt = f"""You are an expert in operations research and mathematical optimization. You need to define a NEW constraint type for the following constraint.

CONSTRAINT:
Constraint Expression: {constraint_text}
Source Information: {source_info}

TASK:
Create a new constraint type definition for this constraint.

RESPONSE FORMAT:
{{
  "type": "concise_name_for_the_new_type",
  "description": "detailed_description_explaining_when_this_constraint_type_is_used_and_what_it_represents",
  "example": "mathematical_example_showing_the_form_of_this_constraint_type"
}}

IMPORTANT GUIDELINES FOR CREATING NEW TYPES:
1. ONLY create a new type if the constraint truly does not fit any existing type.
2. Make the type name concise but descriptive.
3. Make the description detailed and specific, explaining when this type applies.
4. Provide a clear mathematical example of this constraint type.
5. Ensure the new type is general enough to potentially apply to other similar constraints in the future.

Respond ONLY with a valid JSON object in the format above. No other text."""

    return prompt

def call_llm(prompt: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    """
    Call the LLM with the given prompt and parse the response
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies mathematical optimization constraints."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        # Try to parse as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not extract JSON from LLM response")
                
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Return a default classification
        return {"type": "Unclassified", "dimensionality": "unknown"}

def classify_constraint(constraint_text: str, source_info: str, constraint_pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classify a constraint using the LLM and existing constraint pool
    """
    # Create prompt for classification
    prompt = create_classification_prompt(constraint_text, source_info, constraint_pool)
    
    # Call LLM for classification
    result = call_llm(prompt)
    
    # Check if we need to create a new type
    existing_type_names = [ct["type"] for ct in constraint_pool]
    
    if result.get("type") in existing_type_names:
        # Match found in existing types
        return {
            "type": result["type"],
            "dimensionality": result.get("dimensionality", "unknown"),
            "new_type_created": False
        }
    elif "description" in result and "example" in result:
        # New type to be created
        return {
            "type": result["type"],
            "description": result["description"],
            "example": result["example"],
            "dimensionality": result.get("dimensionality", "unknown"),
            "new_type_created": True
        }
    else:
        # Unclassified or error case
        return {
            "type": "Unclassified",
            "dimensionality": "unknown",
            "new_type_created": False
        }

def process_item(item: Dict[str, Any], idx: int, shared_constraint_pool: List[Dict[str, Any]], 
                constraints_file: str, log_file: str, pool_lock) -> Dict[str, Any]:
    """
    Process a single item (with all its constraints)
    """
    print(f"\nPROCESSING ITEM {idx+1}")
    print("-" * 40)
    
    # Get a snapshot of the constraint pool for this item processing
    constraint_pool_snapshot = get_constraint_pool_snapshot(shared_constraint_pool, pool_lock)
    
    # Process each atomic information entry in the item
    for atomic_idx, atomic_info in enumerate(item.get("atomic_information", [])):
        # Extract constraints from related_constraints
        related_constraints = atomic_info.get("related_constraints", [])
        
        # Add type and dimensionality fields to each constraint
        for constraint_idx, constraint_text in enumerate(related_constraints):
            source_info = atomic_info.get("info", "")
            
            print(f"Atomic Info #{atomic_idx+1}, Constraint #{constraint_idx+1}:")
            print(f"  Text: {constraint_text}")
            print(f"  Source: {source_info}")
            
            # Classify the constraint using the snapshot
            classification_result = classify_constraint(constraint_text, source_info, constraint_pool_snapshot)
            
            # Add classification to the constraint
            atomic_info.setdefault("constraints_with_types", []).append({
                "constraint": constraint_text,
                "type": classification_result["type"],
                "dimensionality": classification_result["dimensionality"]
            })
            
            print(f"  Classification: {classification_result['type']}")
            print(f"  Dimensionality: {classification_result['dimensionality']}")
            
            # If a new type was created, add it to the shared constraint pool and log it
            if classification_result.get("new_type_created", False):
                with pool_lock:
                    new_type_entry = {
                        "type": classification_result["type"],
                        "description": classification_result["description"],
                        "example": classification_result["example"]
                    }
                    shared_constraint_pool.append(new_type_entry)
                    print(f"  *** NEW TYPE CREATED: '{classification_result['type']}' ***")
                    
                    # Log the new constraint type
                    log_new_constraint_type(
                        problem_index=item.get("idx", idx),
                        constraint_index=constraint_idx,
                        constraint_text=constraint_text,
                        new_type=classification_result["type"],
                        description=classification_result["description"],
                        example=classification_result["example"],
                        log_file=log_file
                    )
                    
                    # Save the updated constraint pool immediately
                    save_constraint_pool(shared_constraint_pool, constraints_file)
            
            print("-" * 40)
    
    return item

def main():
    if not API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    # File paths
    constraints_file = os.path.join(DATA_DIR, "constraints.json")
    dataset_file = os.path.join(DATA_DIR, "1-info_and_constraints_filtered.json")
    output_file = os.path.join(DATA_DIR, "3-with_constraint_types.json")
    log_file = os.path.join(DATA_DIR, "new_constraint_types_log.json")
    progress_file = os.path.join(DATA_DIR, "classify_constraints_progress.json")
    
    # Load data
    constraint_pool = load_existing_constraints(constraints_file)
    dataset = load_dataset(dataset_file)
    
    # Load progress for resuming
    processed_items = load_progress(progress_file)
    
    # Filter out already processed items
    remaining_dataset = [item for idx, item in enumerate(dataset) if idx not in processed_items]
    
    print("=" * 80)
    print("STARTING CONSTRAINT CLASSIFICATION PROCESS")
    print(f"Total items: {len(dataset)}, Remaining items: {len(remaining_dataset)}")
    print("=" * 80)
    
    # Counter for new constraint types added
    new_types_added = 0
    total_constraints_processed = 0
    batch_size = 128
    processed_indices = []
    
    # Process items in parallel
    MAX_WORKERS = 16
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all remaining items for processing
        future_to_index = {
            executor.submit(process_item, item, idx, constraint_pool, constraints_file, log_file, constraint_pool_lock): idx
            for idx, item in enumerate(dataset) 
            if idx not in processed_items
        }
        
        # Process completed items
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                # Get the processed item
                processed_item = future.result()
                
                # Update the original dataset with the processed item
                dataset[idx] = processed_item
                
                # Mark item as processed
                processed_items.add(idx)
                processed_indices.append(idx)
                
                # Count constraints
                for atomic_info in processed_item.get("atomic_information", []):
                    total_constraints_processed += len(atomic_info.get("constraints_with_types", []))
                
                # Save progress
                save_progress(processed_items, progress_file)
                
                # Batch save every 16 items
                if len(processed_indices) >= batch_size:
                    save_dataset_batch(dataset, output_file, processed_indices)
                    processed_indices.clear()
                    print(f"PROGRESS UPDATE: Saved batch of {batch_size} items, Total constraints processed: {total_constraints_processed}")
                else:
                    print(f"PROGRESS UPDATE: Processed item {idx+1}, Total constraints processed: {total_constraints_processed}")
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
    
    # Save any remaining items
    if processed_indices:
        save_dataset_batch(dataset, output_file, processed_indices)
        print(f"PROGRESS UPDATE: Saved final batch of {len(processed_indices)} items")
    
    # Count new constraint types added
    current_constraint_pool = load_existing_constraints(constraints_file)
    new_types_added = len(current_constraint_pool) - len(constraint_pool)
    
    print("\n" + "=" * 80)
    print("CONSTRAINT CLASSIFICATION COMPLETED")
    print("=" * 80)
    print(f"Total constraints processed: {total_constraints_processed}")
    print(f"Approximate new constraint types added: {new_types_added}")
    print(f"Results saved to {output_file}")
    print(f"Updated constraint types saved to {constraints_file}")
    print(f"New constraint type log saved to {log_file}")
    print(f"Progress saved to {progress_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()