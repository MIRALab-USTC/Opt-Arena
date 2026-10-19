#!/usr/bin/env python3
"""
generate_scenarios.py

Use GPT-4o-mini to generate industry scenario labels for optimization problems.
Input file: data/merged_shuffled_0401_with_difficulties.json
Output file: data/merged_shuffled_0401_with_difficulties_with_scenarios.json

Dependency: openai
"""
import json
import os
import openai
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

# API configuration
API_KEY = os.getenv("OPT_ARENA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("OPT_ARENA_BASE_URL", "https://yunwu.ai/v1")
LLM_MODEL = os.getenv("OPT_ARENA_MODEL", "gpt-4o-mini")

# Parallel processing configuration
MAX_WORKERS = 16  # Maximum concurrent worker threads
BATCH_SIZE = 100  # Batch size
WRITE_INTERVAL = 16  # Number of batches between writes

def ask_llm(prompt):
    """
    Call the language model API
    """
    if not API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [
        {"role": "system", "content": "You are an expert in operations research and optimization problems. Your task is to categorize optimization problems according to the industry or sector they belong to."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL, 
            messages=messages, 
            temperature=0.1, 
            max_tokens=20
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def generate_scenario(question_text):
    """
    Use GPT-4o-mini to generate a scenario field for the given problem
    """
    prompt = f"""
Based on the following optimization problem description, select the most appropriate industry or sector from the list below:

{question_text}

Options:
Energy and Environment
Finance and Insurance
Education
Transportation
Logistics and Supply Chain Management
Manufacturing
Healthcare
Retail
Non-Profit and Government  
Technology  
Agriculture
Hospitality 
Sports 
Military and Defense 
Entertainment 
Telecommunications
Real Estate
Pharmaceuticals  
Aerospace
Automotive 
Construction
Government

Please respond with only the industry name that best fits the problem. For example: "Manufacturing"
"""

    try:
        scenario = ask_llm(prompt)
        return scenario
    except Exception as e:
        print(f"Error generating scenario: {e}")
        return None

def process_item(item):
    """
    Process one problem item
    """
    if 'scenario' not in item:
        question_text = item.get('question', '')
        if question_text:
            scenario = generate_scenario(question_text)
            if scenario:
                item['scenario'] = scenario
                return item
            else:
                print("Failed to generate scenario")
                return item
        else:
            print("No question text found")
            return item
    else:
        return item

def process_batch(batch_data, batch_index, original_data):
    """
    Process one batch of data
    """
    batch_results = []
    print(f"Processing batch {batch_index}, data size: {len(batch_data)}")
    for i, (original_index, item) in enumerate(batch_data):
        try:
            result = process_item(item)
            batch_results.append((original_index, result))
            print(f"Item {original_index} processing completed")
        except Exception as e:
            # Skip a problem if its model call fails
            print(f"Item {original_index} processing failed, skipping: {e}")
            batch_results.append((original_index, item))  # Return the original item instead of stopping
    return batch_index, batch_results

def save_progress(processed_indices, filepath):
    """
    Save only processed problem indices as progress
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(list(processed_indices), f, ensure_ascii=False, indent=2)

def load_progress(filepath):
    """
    Load processing progress
    """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_data(data, output_path):
    """
    Save processed results to a file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated JSON saved to: {output_path}")

def save_failed_indices(data, file_path):
    """
    Save indices missing a scenario field to JSON
    """
    failed_indices = []
    for i, item in enumerate(data):
        if 'scenario' not in item:
            failed_indices.append(i)
    
    failed_file_path = file_path.parent / (file_path.stem + '_failed_indices.json')
    with open(failed_file_path, 'w', encoding='utf-8') as f:
        json.dump(failed_indices, f, indent=2, ensure_ascii=False)
    print(f"Failed indices saved to: {failed_file_path}")
    print(f"Total failed items: {len(failed_indices)}")

def process_json_file(file_path):
    """
    Add scenario fields to a JSON file in parallel with resume support
    """
    output_path = file_path.parent / (file_path.stem + '_with_scenarios' + file_path.suffix)
    progress_file = file_path.parent / (file_path.stem + '_scenarios_progress.json')
    
    # Load input data
    with open(file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Check for a progress file to resume processing
    processed_indices = load_progress(progress_file)
    
    if processed_indices:
        print(f"Found progress file, {len(processed_indices)} items already processed, continuing...")
    
    # Find unprocessed items
    items_to_process = []
    for i, item in enumerate(original_data):
        if i not in processed_indices:
            items_to_process.append((i, item))
    
    print(f"Total data items: {len(original_data)}")
    print(f"Items to process: {len(items_to_process)}")
    
    if not items_to_process:
        print("All data has been processed")
        # Save the final result even if processing is already complete
        save_data(original_data, output_path)
        # Save failed indices
        save_failed_indices(original_data, file_path)
        
        # Remove the progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Progress file removed")
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
                    if 'scenario' in result and result['scenario']:
                        original_data[original_index] = result
                
                # Save all processed indices, including failures
                if batch_processed_indices:  # Save progress only when items were processed
                    save_progress(processed_indices, progress_file)
                    print(f"Progress saved for batch {batch_index} to {progress_file}, processed {len(batch_processed_indices)} items")
                
                # Write output every WRITE_INTERVAL batches
                write_counter += 1
                if write_counter % WRITE_INTERVAL == 0 or batch_index >= len(batches) - 1:
                    save_data(original_data, output_path)
                    print(f"Output file updated {output_path} (processed {write_counter} batches)")
                else:
                    print(f"Batch {batch_index} completed, waiting to write...")
                    
            except Exception as e:
                print(f"Error processing batch {batch_index}: {e}")
                # Save progress even after an error
                save_progress(processed_indices, progress_file)
    
    # Save final results
    save_data(original_data, output_path)
    # Save failed indices
    save_failed_indices(original_data, file_path)
    
    # Remove the progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("Progress file removed")

if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    # Select the JSON file to process
    json_file_path = Path(os.path.join(DATA_DIR, "merged_shuffled_0401_with_difficulties.json"))
    
    # Process the file
    process_json_file(json_file_path)