#!/usr/bin/env python3
"""
generate_atomic_information.py

Use GPT-4o-mini to extract atomic information and related constraints from problems and models.
Input file: data/OptMATH-Train-100k_converted.json
Output file: data/1-info_and_constraints.json

Dependency: openai
"""
import json
import os
import openai
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

API_KEY = os.getenv("OPT_ARENA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("OPT_ARENA_BASE_URL", "https://yunwu.ai/v1")
LLM_MODEL = os.getenv("OPT_ARENA_MODEL", "gpt-4o-mini")
INPUT_FILE = os.path.join(DATA_DIR, "OptMATH-Train-100k_converted.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "1-info_and_constraints.json")
BATCH_SIZE = 10  # Adjustable
SLEEP_BETWEEN_BATCH = 2  # Avoid rate limits
MAX_WORKERS = 16  # Concurrent worker threads

PROMPT_TEMPLATE = '''You are a professional operations research and mathematical modeling expert.

Your task is to extract fine-grained *atomic information* from a given optimization problem ("question") and match them with constraints in a provided mathematical model ("modeling").

### Requirements
1. **Atomic Information Extraction**
   - Split the question into minimal, indivisible *atomic information units*.
   - Each atomic information should describe one specific modeling aspect (e.g., a condition, limitation, relationship, or requirement).
   - Avoid merging multiple ideas into one; keep them as fine-grained as possible.
   - Output each atomic information as a concise, complete English sentence.
   - Only extract information related to constraints, do not extract information about the objective function.

2. **Constraint Matching**
   - From the provided modeling, identify which constraint(s) correspond to each atomic information.
   - Constraints should be quoted or summarized clearly.
   - One atomic information can match multiple constraints, and one constraint can correspond to multiple atomic information units.

3. **Output Format**
   Use the following JSON format strictly:
   [
     {{
       "info": "<atomic information 1>",
       "related_constraints": ["a", "b", "c"]
     }},
     {{
       "info": "<atomic information 2>",
       "related_constraints": ["..."]
     }}
   ]

4. **Notes**
   - Focus only on the logical mapping between the question and the model constraints.
   - Do NOT extract information related to the objective function.
   - Do NOT explain your reasoning or add commentary.

## Example
### Input
Question:
A company produces two products A and B. Each requires machine time and labor. 
Machine time available is 40 hours, labor available is 60 hours. 
Profit per unit: 3 for A, 4 for B. 
The company wants to maximize total profit.

Modeling:
max z = 3x1 + 4x2
s.t.
  2x1 + x2 ≤ 40
  x1 + 3x2 ≤ 60
  x1, x2 ≥ 0   

### Output
[
     {{
       "info": "Available machine time is 40 hours.",
       "related_constraints": ["2x1 + x2 ≤ 40"]
     }},
     {{
       "info": "Available labor is 60 hours.",
       "related_constraints": ["x1 + 3x2 ≤ 60"]
     }}
]

## Now process the following problem:
### Input
Question:
{question}

Modeling:
{modeling}
'''

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def ask_llm(prompt):
    if not API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [
        {"role": "system", "content": "You are an operations research modeling expert who only returns structured JSON."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.5, max_tokens=2000)
    text = resp.choices[0].message.content.strip()
    import re
    try:
        return json.loads(text)
    except Exception as e1:
        # Try to extract JSON
        m = re.search(r'\[.*\]', text, re.S)
        json_str = m.group(0) if m else text
        try:
            return json.loads(json_str)
        except Exception as e2:
            # Double single backslashes and parse again
            fixed = json_str.replace('\\', '\\\\')
            try:
                return json.loads(fixed)
            except Exception as e3:
                raise RuntimeError(f"Could not parse LLM response: {text}\nInitial error: {e1}\nExtraction error: {e2}\nBackslash correction error: {e3}")

def process_item(item):
    """Process one data item"""
    prompt = PROMPT_TEMPLATE.format(
        question=item.get("question", ""),
        modeling=item.get("modeling", "")
    )
    try:
        result = ask_llm(prompt)
        
        # If the result is a dictionary, extract atomic_information
        if isinstance(result, dict):
            if "atomic_information" in result:
                result = result["atomic_information"]
            else:
                # It may be wrapped under another field; try the first value
                values = list(result.values())
                if values and isinstance(values[0], list):
                    result = values[0]
                else:
                    result = []

        # At this point result should be a list
        if isinstance(result, list):
            validated_result = []
            for r in result:
                if isinstance(r, dict) and 'info' in r and 'related_constraints' in r:
                    validated_result.append(r)
                else:
                    print(f"Warning: atomic information item has an unexpected structure: {r}")
            return validated_result
        else:
            print(f"Warning: LLM result is not a list; actual type: {type(result)}, content: {result}")
            return []
    except Exception as e:
        print(f"Failed to process problem: {e}")
        return []

def process_batch(batch_data, batch_index, original_data):
    """Call the LLM once per problem and return results aligned with the batch"""
    batch_results = []
    print(f"Processing batch {batch_index}, size: {len(batch_data)}")
    for i, (original_index, item) in enumerate(batch_data):
        try:
            result = process_item(item)
            batch_results.append((original_index, result))
            print(f"Problem {original_index} completed")
        except Exception as e:
            # Skip a problem if its model call or response parsing fails
            print(f"Problem {original_index} failed; skipping: {e}")
            batch_results.append((original_index, []))  # Append an empty result instead of stopping
    return batch_index, batch_results

def save_progress(processed_indices, filepath):
    """Save only processed problem indices as progress"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(list(processed_indices), f, ensure_ascii=False, indent=2)

def load_progress(filepath):
    """Load processing progress"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def main():
    if not API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    input_path = INPUT_FILE
    output_path = OUTPUT_FILE
    WRITE_INTERVAL = 10  # Write output every 10 batches
    
    # Load input data
    original_data = load_data(input_path)
    
    # Check for a progress file to resume processing
    progress_file = input_path.replace('.json', '_atomic_info_progress.json')
    processed_indices = load_progress(progress_file)
    
    if processed_indices:
        print(f"Found progress file; {len(processed_indices)} problems processed. Resuming...")
    
    # Find unprocessed items
    items_to_process = []
    for i, item in enumerate(original_data):
        if i not in processed_indices:
            items_to_process.append((i, item))
    
    print(f"Total data items: {len(original_data)}")
    print(f"Items remaining: {len(items_to_process)}")
    
    if not items_to_process:
        print("All data has been processed")
        # Save the final result even if processing is already complete
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(f"Saved updated atomic information to {output_path}")
        
        # Remove the progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Removed temporary progress file")
        return
    
    # Split work into BATCH_SIZE batches
    batches = []
    for idx in range(0, len(items_to_process), BATCH_SIZE):
        batch_items = items_to_process[idx:idx+BATCH_SIZE]
        batch_index = idx // BATCH_SIZE  # Calculate the current batch index
        batches.append((batch_index, batch_items))
    
    print(f"Total batches: {len(batches)}")
    
    # Process batches in a thread pool
    results = [None] * len(batches)  # Preallocate the result list
    write_counter = 0  # Initialize the write counter
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batch tasks
        future_to_batch_index = {
            executor.submit(process_batch, batch_items, batch_index, original_data): batch_index 
            for batch_index, batch_items in batches
        }
        
        # Collect processed results
        for future in as_completed(future_to_batch_index):
            try:
                batch_index, batch_results = future.result()
                results[batch_index] = batch_results
                
                # Update progress for both successful and failed items
                batch_processed_indices = []  # Record this batch's indices, including failures
                for original_index, result in batch_results:
                    batch_processed_indices.append(original_index)
                    processed_indices.add(original_index)
                    
                    # Update data only for successful problems
                    if result:  # A nonempty result indicates success
                        original_data[original_index]["atomic_information"] = result
                    else:
                        # Record failures in progress without updating data
                        print(f"Problem {original_index} failed; recorded in progress but not written to output")
                
                # Save all processed indices, including failures
                if batch_processed_indices:  # Save progress only when items were processed
                    save_progress(processed_indices, progress_file)
                    print(f"Saved batch {batch_index} progress to {progress_file}; processed {len(batch_processed_indices)} problems")
                
                # Write output every WRITE_INTERVAL batches
                write_counter += 1
                if write_counter % WRITE_INTERVAL == 0 or batch_index >= len(batches) - 1:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(original_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated output {output_path} after {write_counter} batches")
                else:
                    print(f"Batch {batch_index} completed; output write deferred")
                
            except Exception as e:
                print(f"Error processing batch {batch_index}: {e}")
                # Save progress even after an error
                save_progress(processed_indices, progress_file)
    
    # Save final results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    print(f"Saved updated atomic information to {output_path}")
    
    # Remove the progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("Removed temporary progress file")

if __name__ == "__main__":
    main()