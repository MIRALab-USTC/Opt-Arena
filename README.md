# OptMATH: Optimization Problem Analysis and Generation

This repository contains research code for processing natural-language mathematical optimization problems. It provides an LLM-assisted difficulty analysis interface, a problem generation interface, and scripts for building annotations such as problem types, constraint types, atomic information, and scenario labels. Running the LLM workflows can incur provider charges.

The repository contains reference JSON catalogs, but it does not contain the source problem dataset or every intermediate file used by the legacy scripts. A complete data processing run requires those inputs.

## Key features

- Analyze one problem or a JSONL collection: derive or accept a mathematical model, extract atomic information, classify problem and constraint types, match kernels, and calculate difficulty scores.
- Generate problems by requested type counts or from kernels observed in prior analysis results.
- Rebuild selected annotations with standalone scripts for atomic information, classifications, merged atom information, scenario labels, and difficulty scores.
- Load the supplied catalogs of 56 problem types, 50 scored constraint types, atomic information groups, and kernel combinations. These counts describe files, not evaluation results.

## Repository structure

| Path | Purpose |
| --- | --- |
| problem_difficulty_interface.py | Single-problem and JSONL difficulty analysis entry point. |
| problem_generation_interface.py | Manual and similarity-based problem generation entry point. |
| generate_problem_manual_mode.py | Bundled prompt templates and a standalone legacy generation workflow. |
| problem_type.json | Problem type catalog. |
| constraints_with_difficulty_scores-train.json | Constraint type catalog with difficulty scores. |
| atomic_info_all-2.json, classified_atomic_info_groups.json | Atomic information reference groups. |
| kernel_combinations.json | Kernel and combination frequency reference data. |
| generate_info_and_constraints.py, classify_problems.py, classify_optmath_constraints.py | Legacy LLM annotation stages. |
| multiprocess_add_atom_info.py, calculate_constraint_difficulty_scores.py, calculate_optmath_problem_difficulty.py, delete_problem.py | Legacy processing and scoring utilities. |
| graph/ | Numbered annotation and scenario stages. The first two mirror root scripts except for relative path setup. |
| DIFFICULTY_INTERFACE.md, GENERATION_INTERFACE.md | Interface details and record schemas. |
| .env.example, .gitignore, requirements.txt, CITATION.cff | Environment example, publication exclusions, dependencies, and citation metadata. |
| LICENSE, DATA_LICENSE.md | Code and documentation license; separate license for five reference JSON catalogs. |

The five root JSON files are reference inputs. Place source datasets, generated JSON/JSONL, logs, progress markers, model weights, and credentials in ignored local directories.

## Installation

The pinned dependencies in requirements.txt match the tested vllm_eval environment with Python 3.10.20. From the repository root, activate that environment and install the direct dependencies if needed:

~~~bash
conda activate vllm_eval
python -m pip install -r requirements.txt
~~~

A fresh Python 3.10 environment can also install the same requirements.

Direct Python dependencies are openai, pandas, and ijson. The main interfaces use openai; the bundled standalone generation module also uses pandas and ijson. The repository does not import vllm. Select the backend model and provider separately. Gurobi and its license are needed only to execute generated Gurobi Python code. Generating its text does not require gurobipy.

## Quick start

This offline command checks that the bundled reference catalogs load without an API key:

~~~bash
python - <<'PY'
from problem_difficulty_interface import load_analysis_resources
resources = load_analysis_resources()
print(len(resources["problem_types"]), "problem types")
print(len(resources["constraint_catalog"]), "constraint types")
PY
~~~

To run an analysis, configure an API key and compatible model:

~~~bash
export OPT_ARENA_API_KEY="your-key"
export OPT_ARENA_MODEL="your-model-name"
# Set OPT_ARENA_BASE_URL if your provider requires a custom endpoint.
mkdir -p outputs
python problem_difficulty_interface.py \
  --question "A factory must allocate limited machine hours to products to maximize profit." \
  --output outputs/example_difficulty.json
~~~

The analysis command invokes the model and has not been exercised in the publication checks.

## Usage

### Difficulty analysis

Run "python problem_difficulty_interface.py --help" for all arguments. Supplying --modeling or --modeling-file skips the formulation call; the remaining analysis still calls the LLM. For batch input, use --input-jsonl path/to/problems.jsonl --output-jsonl outputs/difficulty.jsonl. Input records must include a question field; [DIFFICULTY_INTERFACE.md](DIFFICULTY_INTERFACE.md) documents accepted fields and output structure. The interface writes a statistics file alongside JSONL output unless --stats-output is set.

### Problem generation

The generation interface uses the bundled generate_problem_manual_mode.py prompt module by default. The module defines get_prompts_direct(problem_type) and get_prompts_complex(problem_type, mapped), returning prompt templates with p, m, c, and nl keys. Override it with --prompt-module-file only if you have another compatible implementation.

After configuring the API:

~~~bash
mkdir -p outputs
python problem_generation_interface.py \
  --mode manual \
  --type-counts '{"Production Planning Problem": 1}' \
  --output-jsonl outputs/generated.jsonl
~~~

Use --mode similar --analysis-jsonl outputs/difficulty.jsonl to sample from kernels in prior difficulty results. problem_types_atomic_info_statistics.json is optional and absent here; if available, supply it with --pt-stats-file. See [GENERATION_INTERFACE.md](GENERATION_INTERFACE.md) for generation modes and output schema. This command requires a working API account and makes paid model requests.

### Legacy data pipeline

The older annotation scripts use OPTMATH_DATA_DIR for missing source and intermediate files, defaulting to ./data. The standalone generate_problem_manual_mode.py script uses the bundled catalogs, an optional statistics file under ./data, and the ignored ./outputs directory by default. They preserve the original batch parameters and filenames. Their specific input JSON schemas are required; this checkout alone is not a turnkey pipeline. In particular, OptMATH-Train-100k_converted.json, some constraint catalogs, and several later intermediate files are absent. Set OPTMATH_DATA_DIR to a local directory holding those files. Keep it outside Git or under the ignored data/ directory.

The numbered scripts in graph/ are related stages. Their input filenames show that additional merging or reclassification occurred between some stages; no missing transformation has been inferred or recreated here. Inspect the script constants and input schema before running a stage on new data. Some stages make many concurrent paid API calls.

## Configuration

| Variable or option | Meaning |
| --- | --- |
| OPT_ARENA_API_KEY | Preferred API credential. Interfaces and legacy scripts also accept OPENAI_API_KEY or DEEPSEEK_API_KEY. |
| OPT_ARENA_BASE_URL | OpenAI-compatible endpoint. Main interfaces use the client default if unset. Legacy annotation scripts retain their previous endpoint as default. |
| OPT_ARENA_MODEL | Model name; defaults to gpt-4o-mini. |
| OPTMATH_DATA_DIR | Local legacy data directory; defaults to ./data relative to the repository. |
| EVO_OPT_ARENA_DIR | Optional directory for alternate generation prompts and reference catalogs; defaults to this repository. |
| OPTMATH_PT_STATS_FILE | Optional statistics file for the standalone generator; defaults to data/problem_types_atomic_info_statistics.json. |
| OPTMATH_EVALUATION_STATS_FILE | Evaluation statistics for the standalone generator in automatic mode; defaults to data/problem_type_analysis.json. |
| --*-file options | Override catalog, dataset, or prompt-module paths in the main interfaces. |

Use [.env.example](.env.example) as a list of variables to configure. Scripts read environment variables; they do not load an .env file automatically. Never commit real keys.

## Data and reproducibility

The bundled catalogs support an offline resource-loading check and analysis metadata preparation. Full experiments require absent source problems and intermediate files, an API account, and the original model/provider configuration. LLM responses can differ across providers, versions, and sampling runs. The generation interface supports --seed for local kernel and atom sampling; it does not fix remote model outputs.

The project owner has confirmed that the five bundled JSON reference catalogs may be redistributed under CC BY 4.0. Their scope, attribution, and translation changes are recorded in [DATA_LICENSE.md](DATA_LICENSE.md). The provenance and reuse terms for any source dataset added later should be documented separately. No source download link or fixed dataset split can be confirmed from this checkout.

## Citation

If you use this repository, please cite [Opt-Arena](https://openreview.net/forum?id=HKMOHcifiP). The author names and venue below follow the citation information provided by the project owner. A machine-readable citation is also available in [CITATION.cff](CITATION.cff).

~~~bibtex
@inproceedings{anonymous2026optarena,
  title={Opt-Arena: Evaluating, Selecting, and Generating Optimization Modeling Data via Tripartite Graphs},
  author={Yian Xu and Xiongwei Han and Jie Wang and Haoyang Liu and Yuyang Cai and Boxuan Niu and Mingxuan Yuan},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=HKMOHcifiP}
}
~~~

## License

The source code and documentation are licensed under the [MIT License](LICENSE). The five bundled reference JSON catalogs are licensed separately under [CC BY 4.0](DATA_LICENSE.md). These terms do not cover third-party dependencies, source datasets absent from this repository, or generated outputs.
