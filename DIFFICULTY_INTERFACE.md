# OptMATH Difficulty Analysis Interface

The default reference directory is this repository. The optional fallback file constraints_with_difficulty.json is not bundled; analysis can use the included scored catalog alone.

## Analysis workflow

The analyze_problem_difficulty(question, modeling=None) function:

1. Generates a mathematical model if modeling is omitted.
2. Extracts atomic information and related constraints from the question and model.
3. Classifies the problem type with problem_type.json.
4. Classifies each constraint with constraints_with_difficulty_scores-train.json.
5. Matches generic atom_info by problem type and constraint types using atomic_info_all-2.json.
6. Matches a kernel using kernel_combinations.json.
7. Calculates constraint, atomic information, and overall problem difficulty.

The scoring rule is:

~~~text
constraint difficulty = base type difficulty + 0.25 if multidimensional
atomic difficulty = sum of related constraint difficulties
problem difficulty = maximum atomic difficulty
~~~

The interface prefers difficulty_scores.difficulty_score from the scored catalog. If an optional fallback catalog contains a type absent from the scored catalog, it calculates its base score from that type's difficulty fields, using the rules in calculate_constraint_difficulty_scores.py.

## Python usage

~~~python
from problem_difficulty_interface import analyze_problem_difficulty

result = analyze_problem_difficulty(
    question="A company assigns workers to tasks ...",
    modeling=None,  # Supply existing modeling to skip the formulation call.
)
print(result["problem_type"])
print(result["difficulty"])
~~~

This function calls an LLM even when modeling is supplied.

## Command line usage

Set a credential and, when needed, a compatible endpoint:

~~~bash
export OPT_ARENA_API_KEY="your-api-key"
export OPT_ARENA_BASE_URL="https://your-openai-compatible-endpoint.example/v1"
export OPT_ARENA_MODEL="gpt-4o-mini"
mkdir -p outputs
python problem_difficulty_interface.py \
  --question "A factory allocates limited machine hours to maximize profit." \
  --output outputs/difficulty_result.json
~~~

To provide an existing formulation, add --modeling-file path/to/modeling.txt. Use python problem_difficulty_interface.py --help for all options.

## JSONL batch mode

Each input line is one JSON object. The interface accepts question, problem, problem_text, problem_description, natural_language_desc, generation.problem_description, or generation.natural_language_desc as the problem text. Modeling may be in modeling, abstract_modeling, model, or generation.abstract_modeling.

Example input:

~~~json
{"idx": 1, "question": "A factory allocates limited machine hours.", "modeling": "Maximize total profit subject to machine-hour limits."}
{"idx": 2, "question": "A warehouse assigns deliveries to vehicles."}
~~~

Run a batch with:

~~~bash
mkdir -p outputs
python problem_difficulty_interface.py \
  --input-jsonl path/to/problems.jsonl \
  --output-jsonl outputs/difficulty_results.jsonl \
  --limit 2
~~~

A .jsonl path passed to --question-file is also treated as batch input when no single-problem arguments are given. Use --include-source-metadata to include the input index and source ID. By default, a statistics file is written at outputs/difficulty_results.jsonl.stats.json; --stats-output overrides that path.

The batch statistics include problem and error counts, type distributions, atomic information and constraint type counts, kernel counts, average difficulty, and dataset overlap. Dataset overlap is total atomic information occurrences divided by the number of distinct atomic information types across all processed records. It is not the mean of per-problem overlaps.

## Output

Single-problem output and each successful JSONL result include the question, modeling, problem type, kernel match, atomic information with typed constraints, constraint difficulties, overlap statistics, overall difficulty, and a description of the difficulty rule. The exact fields are defined by problem_difficulty_interface.py. No numerical values in this document are reported experimental results.
