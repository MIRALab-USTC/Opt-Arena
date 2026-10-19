# Opt-Arena Problem Generation Interface

The prompt module generate_problem_manual_mode.py is included in this repository. The main entry point is problem_generation_interface.py. It uses the prompt module for four stages: problem description, mathematical modeling, Gurobi Python code, and a natural-language rewrite. The code and rewrite stages can be skipped.

## Configuration

Set a credential and, when needed, an OpenAI-compatible endpoint:

~~~bash
export OPT_ARENA_API_KEY="your-api-key"
export OPT_ARENA_BASE_URL="https://your-openai-compatible-endpoint.example/v1"
export OPT_ARENA_MODEL="gpt-4o-mini"
~~~

The interface also accepts the DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, and DEEPSEEK_MODEL variables. It reads problem_type.json, kernel_combinations.json, atomic_info_all-2.json, and constraints_with_difficulty_scores-train.json from this repository by default. The optional problem_types_atomic_info_statistics.json is absent; provide it with --pt-stats-file if available.

## Manual generation

Provide problem types and requested counts as JSON:

~~~bash
mkdir -p outputs
python problem_generation_interface.py \
  --mode manual \
  --type-counts '{"Production Planning Problem": 1}' \
  --output-jsonl outputs/generated_manual.jsonl
~~~

Alternatively, supply --type-counts-file path/to/type_counts.json. A type found in the kernel or atomic information catalogs uses Mode A; another type uses Mode B (Direct). Mode A samples a kernel and supplements it with atomic information from the same problem type.

## Similar generation

First analyze an input JSONL file with problem_difficulty_interface.py. Then use the resulting problem_type and kernel.atom_infos fields:

~~~bash
python problem_generation_interface.py \
  --mode similar \
  --analysis-jsonl outputs/difficulty_results.jsonl \
  --similar-count 2 \
  --output-jsonl outputs/generated_similar.jsonl
~~~

The interface aggregates records by (problem_type, kernel) and generates the requested count per observed pair. To specify counts directly, supply --similar-plan-file with a JSON list of objects containing problem_type, kernel, and count.

## Output and options

Each JSONL record contains problem_type, mode, kernel, selected_atomic_infos, source, observed_count, and generation. The generation object contains problem_description, abstract_modeling, gurobi_code, natural_language_desc, and timestamp. Code and natural-language fields may be null when skipped.

Useful options:

~~~text
--target-atomic-count 6   Final number of atomic information items per problem
--max-workers 1           Concurrent generation workers
--skip-code               Skip Gurobi code generation
--skip-nl                 Skip the natural-language rewrite
--seed 42                 Seed local kernel and atomic information sampling
--prompt-module-file PATH Override the bundled prompt module
~~~

The seed does not make remote model responses deterministic. These commands make paid API requests. Generated Gurobi code is text; executing it separately requires Gurobi and its license.
