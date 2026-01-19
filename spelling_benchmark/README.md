# Spelling Benchmark

A synthetic benchmark to evaluate character-level understanding in language models.

## Motivation

The "Strawberry Problem": LLMs often fail at simple spelling tasks like "How many r's are in strawberry?" because tokenization obscures individual letters. This benchmark rigorously tests whether models can reason about characters within tokens.

## Task Types

The benchmark includes three task types, evenly distributed (333/333/334):

| Task | Example Input | Expected Output |
|------|---------------|-----------------|
| **Count** | How many times does the letter 'r' appear in the word 'strawberry'? | 3 |
| **Index** | What is the 3rd letter of the word 'python'? | t |
| **Reverse** | Reverse the word 'stressed'. | desserts |

## Tokenization Metadata

Each sample includes metadata to analyze performance by tokenization:

```json
{
  "input": "How many times does the letter 'r' appear in the word 'strawberry'?",
  "target": "3",
  "task_type": "count",
  "metadata": {
    "word": "strawberry",
    "char": "r",
    "num_tokens": 3,
    "is_single_token": false
  }
}
```

This enables reporting **Single-Token Accuracy** vs **Multi-Token Accuracy** separately:
- **Single-token words** (e.g., "apple" = 1 token): Tests if the model can decompose atomic tokens into letters
- **Multi-token words** (e.g., "strawberry" = 3 tokens): Tests cross-boundary letter tracking

## Word Sources

Words are sampled 50/50 from:
1. **Common words**: [Google 10,000 English](https://github.com/first20hours/google-10000-english) (filtered to 4-10 chars)
2. **Rare words**: [dwyl/english-words](https://github.com/dwyl/english-words) (~370k words, deduped)

Local copies are stored in `assets/` for reproducibility.

## Usage

### Generate the benchmark

```bash
# Generate 1000 samples (default)
python generate_spelling_benchmark.py

# Custom options
python generate_spelling_benchmark.py --samples 2000 --seed 123 --output path/to/output.jsonl
```

### Run evaluation with lm-eval

```bash
python eval_main.py \
  --checkpoint_paths checkpoints/your_model.pt \
  --tasks spelling_bee \
  --max_samples_log 0  # save all samples for analysis (default: 100)
```

### Analyze results

After running evaluation, analyze results with stratified metrics:

```bash
# From lm-eval results (use --max_samples_log 0 during eval to save all samples)
python -m spelling_benchmark.analyze_results --results_file lm_eval_results/ckpt_spelling_bee.jsonl

# Or run direct inference (useful if you didn't save all samples)
python -m spelling_benchmark.analyze_results --checkpoint checkpoints/model.pt
```

Output includes:
- Overall accuracy
- Single-token vs multi-token accuracy
- Per-task accuracy (count, index, reverse)
- Cross-tabulated task × token type accuracy
- Sample error analysis

### Run tests

```bash
pytest spelling_benchmark/spelling_benchmark_test.py
```

## Files

```
spelling_benchmark/
├── generate_data.py          # Core generation logic (library module)
├── analyze_results.py        # Post-evaluation analysis with stratified metrics
├── spelling_benchmark_test.py # Unit tests
├── spelling_bee.yaml         # lm-evaluation-harness config
├── spelling_bee.jsonl        # Generated benchmark (1000 samples)
└── assets/
    ├── common_words.txt      # Google 10k most common words
    └── full_words.txt        # Full English word list

# In project root:
generate_spelling_benchmark.py  # Runner script
```

## Typical Results

With the default tokenizer (tiktoken cl100k_base):
- ~22% of samples use single-token words
- ~78% of samples use multi-token words

This distribution reflects natural English word length vs tokenizer vocabulary.
