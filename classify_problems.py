import json
import openai
from typing import List, Dict, Any
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("OPTMATH_DATA_DIR", os.path.join(ROOT_DIR, "data"))

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("OPT_ARENA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("OPT_ARENA_BASE_URL", "https://yunwu.ai/v1")
DEEPSEEK_MODEL = os.getenv("OPT_ARENA_MODEL", "gpt-4o-mini")

# Initialize a DeepSeek client per thread
def get_client():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    from openai import OpenAI
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def load_problem_types(file_path: str) -> List[Dict]:
    """Load problem type definitions"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_problem_data(file_path: str) -> List[Dict]:
    """Load problems to classify"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_classification_prompt(question_data: Dict, problem_types: List[Dict]) -> str:
    """Build the classification prompt"""
    # Get the problem description
    question_text = question_data.get('question', '')
    
    # Build available types with names, descriptions, and examples
    type_descriptions = []
    for pt in problem_types:
        type_desc = f"Problem Type: {pt['problem']}\nDescription: {pt['description']}\nExample: {pt['example']}"
        type_descriptions.append(type_desc)
    
    type_list = "\n\n".join(type_descriptions)
    
    prompt = f"""
You are an expert in optimization problem classification. Your task is to classify the following optimization problem into one of the provided categories.

Problem to classify:
{question_text}

Available problem types:
{type_list}

Please select the most appropriate problem type from the list above. Respond ONLY with the problem type name exactly as it appears in the "Problem Type" field above.
If the problem does not fit any of the provided categories, respond with "Unknown".
"""
    
    return prompt

def classify_problem_with_deepseek(question_data: Dict, problem_types: List[Dict], allowed_types: List[str]) -> str:
    """Classify a problem with DeepSeek"""
    prompt = build_classification_prompt(question_data, problem_types)
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in optimization problem classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        # Extract the classification result
        result = response.choices[0].message.content.strip()
        
        # Check that the returned type is allowed
        if result in allowed_types:
            return result
        else:
            print(f"DeepSeek returned a type outside the allowed list: {result}")
            return "Unknown"
    except Exception as e:
        print(f"Error classifying problem with DeepSeek: {e}")
        return "Unknown"

def save_progress(results: List[Dict], output_file: str):
    """Save classification results"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_progress(output_file: str) -> List[Dict]:
    """Load existing classification results"""
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_unknown_problems(unknown_problems: List[Dict], output_file: str):
    """Save problems classified as Unknown"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unknown_problems, f, indent=2, ensure_ascii=False)

def process_single_problem(args):
    """Process one problem in parallel"""
    problem, available_problem_types, allowed_type_names, i, total = args
    idx = problem.get('idx', i)
    
    print(f"Processing problem {i+1}/{total} (New ID: {idx})...")
    
    # Run classification
    predicted_type = classify_problem_with_deepseek(problem, available_problem_types, allowed_type_names)
    
    # Record details for Unknown classifications
    unknown_info = None
    if predicted_type == "Unknown":
        # Build details for an unknown problem
        unknown_info = {
            "idx": idx,
            "question": problem.get('question', ''),
            "generated_type": "Unknown"
        }
    
    # Return results
    result = {
        "idx": idx,
        "predicted_type": predicted_type
    }
    
    # Delay to avoid API rate limits
    time.sleep(0.5)
    
    return result, unknown_info

def main():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing API key. Set OPT_ARENA_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY.")
    # File paths
    problem_data_file = os.path.join(DATA_DIR, "1-info_and_constraints_filtered.json")
    problem_types_file = os.path.join(ROOT_DIR, "problem_type.json")
    output_file = os.path.join(DATA_DIR, "2.1-with_problem_type.json")
    unknown_problems_file = os.path.join(DATA_DIR, "unknown_problems_deepseek.json")
    
    # Load data
    print("Loading problem data...")
    problem_data = load_problem_data(problem_data_file)
    
    print("Loading problem types...")
    problem_types = load_problem_types(problem_types_file)
    
    # Build the list of allowed type names
    allowed_type_names = [pt['problem'] for pt in problem_types]
    allowed_type_names.append("Unknown")  # Include Unknown as an allowed type
    
    print(f"Total problems to classify: {len(problem_data)}")
    print(f"Available problem types: {len(problem_types)}")
    print("Allowed problem types:")
    for type_name in allowed_type_names:
        print(f"  - {type_name}")
    
    # Load existing progress
    results = load_progress(output_file)
    processed_ids = {r['idx'] for r in results}
    
    print(f"Already processed: {len(processed_ids)} problems")
    
    # Select unprocessed problems
    unprocessed_problems = []
    for i, problem in enumerate(problem_data):
        idx = problem.get('idx', i)
        if idx not in processed_ids:
            unprocessed_problems.append((problem, problem_types, allowed_type_names, i, len(problem_data)))
    
    print(f"Problems to process: {len(unprocessed_problems)}")
    
    # Store problems classified as Unknown
    unknown_problems = []
    
    # Process problems with a thread pool
    max_workers = 16  # Reduce concurrency to avoid API limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_index = {executor.submit(process_single_problem, args): args[3] 
                          for args in unprocessed_problems}
        
        # Collect results
        for future in as_completed(future_to_index):
            try:
                result, unknown_info = future.result()
                results.append(result)
                
                # Add Unknown problems to unknown_problems
                if unknown_info:
                    unknown_problems.append(unknown_info)
                
                # Save progress periodically
                if len(results) % 100 == 0:
                    save_progress(results, output_file)
                    print(f"Progress saved. Completed: {len(results)}/{len(problem_data)}")
                    
            except Exception as e:
                print(f"Error processing problem: {e}")
    
    # Save final classification results
    save_progress(results, output_file)
    print(f"Classification completed. Results saved to {output_file}")
    
    # Save problems classified as Unknown
    save_unknown_problems(unknown_problems, unknown_problems_file)
    print(f"Unknown problems saved to {unknown_problems_file}")
    print(f"Total unknown problems: {len(unknown_problems)}")

if __name__ == "__main__":
    main()