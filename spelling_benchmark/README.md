# Spelling Benchmark

A synthetic benchmark to evaluate character-level understanding in language models, designed to test whether spelling bee embeddings improve character-level reasoning.

## Motivation

The "Strawberry Problem": LLMs often fail at simple spelling tasks like "How many r's are in strawberry?" because tokenization obscures individual letters. This benchmark rigorously tests whether models can reason about characters within tokens.

## Task Types

The benchmark includes three task types with few-shot prompting:

| Task | Format | Example |
|------|--------|---------|
| **Count** | `The number of times the letter A occurs in banana is ` | `3` |
| **Index** | `Q: What is the third letter of the word 'banana'? A:` | `n` |
| **Reverse** | `cat reversed is ` | `tac` |

### Default Distribution (n=5000)

- **Count**: 2,450 samples (49%)
- **Index**: 2,450 samples (49%)
- **Reverse**: 100 samples (2%) - reduced due to 0% accuracy on both models

## Few-Shot Examples

Each task type uses 3 few-shot examples:

**Count:**
```
The number of times the letter A occurs in banana is 3
The number of times the letter E occurs in elephant is 2
The number of times the letter O occurs in book is 2
```

**Index:**
```
Q: What is the first letter of the word 'apple'? A:a
Q: What is the third letter of the word 'banana'? A:n
Q: What is the second letter of the word 'cat'? A:a
```

**Reverse:**
```
cat reversed is tac
dog reversed is god
hello reversed is olleh
```

## Results

Evaluated on the 816m parameter models (764m non-embedding parameters):

| Task | Baseline | +Spelling Bee | Delta |
|------|----------|---------------|-------|
| Count | 12.6% | 27.4% | **+14.8%** |
| Index | 10.3% | 16.2% | **+5.9%** |
| Reverse | 0.0% | 0.0% | +0.0% |
| **Overall** | **11.2%** | **21.4%** | **+10.2%** |

**Key findings:**
- Spelling bee embeddings nearly double overall accuracy
- Count task shows largest improvement (+14.8%)
- Reverse task: 0% for both models (outputs forward spelling instead)

## Dataset Construction

### Word Sources
Words are sampled 50/50 from:
1. **Common words**: [Google 10,000 English](https://github.com/first20hours/google-10000-english) (filtered to 4-10 chars)
2. **Rare words**: [dwyl/english-words](https://github.com/dwyl/english-words) (~238k words, deduped)

Local copies are stored in `assets/` for reproducibility.

### Tokenization Metadata

Each sample includes metadata for analysis:

```json
{
  "input": "The number of times the letter R occurs in strawberry is ",
  "target": "3",
  "task_type": "count",
  "metadata": {
    "word": "strawberry",
    "char": "R",
    "num_tokens": 3,
    "is_single_token": false
  }
}
```

### Filtering
- Palindromes are excluded from the reverse task to prevent trivial solutions
- Words filtered to 4-10 characters

## Usage

### Generate the benchmark

```python
from spelling_benchmark.generate_data import generate_dataset

# Generate 5000 samples with custom reverse count
generate_dataset(
    output_path='spelling_benchmark/spelling_bee.jsonl',
    num_samples=5000,
    seed=42,
    reverse_samples=100  # Fixed number of reverse samples
)
```

### Run evaluation with lm-eval

```bash
python eval_main.py \
  --checkpoint_paths checkpoints/baseline.pt checkpoints/spelling_bee.pt \
  --tasks spelling_bee \
  --max_samples_log 0  # Save all samples for per-task analysis
```

### Analyze per-task results

```python
import json

with open('lm_eval_results/model_spelling_bee.jsonl') as f:
    data = json.load(f)

counts = {'count': [0, 0], 'index': [0, 0], 'reverse': [0, 0]}
for sample in data['samples']:
    task_type = sample['doc']['task_type']
    counts[task_type][1] += 1
    if sample.get('exact_match', 0) == 1.0:
        counts[task_type][0] += 1

for task, (correct, total) in counts.items():
    print(f"{task}: {correct}/{total} ({100*correct/total:.1f}%)")
```

### Run tests

```bash
pytest spelling_benchmark/spelling_benchmark_test.py
```

## Files

```
spelling_benchmark/
├── generate_data.py           # Core generation logic
├── spelling_benchmark_test.py # Unit tests
├── spelling_bee.yaml          # lm-evaluation-harness config
├── spelling_bee.jsonl         # Generated benchmark (5000 samples)
└── assets/
    ├── common_words.txt       # Google 10k most common words
    └── full_words.txt         # Full English word list
```
