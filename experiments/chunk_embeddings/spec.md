# Chunk Embeddings: Sparse Memory via Learned Clustering

## Background: Engram Paper

https://www.arxiv.org/pdf/2601.07372

The **Engram** paper introduces "conditional memory" as a new sparsity axis for LLMs, complementing MoE. Key ideas:

- **Problem**: Transformers lack a native primitive for knowledge lookup, forcing them to simulate retrieval through computation
- **Solution**: Engram modernizes N-gram embeddings for O(1) lookup
- **Results**: Scaling to 27B parameters, achieving gains on MMLU (+3.4), BBH (+5.0), HumanEval (+3.0), MATH (+2.4)
- **Mechanism**: Relieves early layers from static reconstruction, freeing attention for global context

### How Engram Works

Engram uses **N-gram hashing** for lookups:
1. Take the last N tokens as context (e.g., N=8)
2. Hash the N-gram to get an index into an embedding table
3. Look up the corresponding embedding and add it to the residual stream

This has limitations:
- **Hash collisions**: Different N-grams may map to the same embedding
- **No generalization**: Slight variations in context get completely different embeddings
- **Fixed N**: Can't capture variable-length patterns

## Our Idea: Embedding + Clustering Instead of Hashing

Instead of hashing N-grams, we propose:

1. **Create embeddings** for context windows using a small embedding model
2. **Cluster** these embeddings to create a finite vocabulary of "context patterns"
3. **Learn sparse embeddings** for each cluster, which get added to the residual stream

### Advantages

- **Semantic similarity**: Similar contexts map to the same cluster (no collision problem)
- **Generalization**: The embedding model captures meaning, not just token sequences
- **Variable context**: The embedding model can handle variable-length context naturally
- **Trainable**: Both the clustering and sparse embeddings can be trained end-to-end

### Overview

**Offline (preprocessing):**
```
Context window (last 32 tokens)
    |
    v
Embedding model (frozen) --> query embedding
    |
    v
Nearest centroid lookup --> cluster_id
    |
    v
Store with training data: (tokens, cluster_ids)
```

**Online (during training):**
```
Input: (token_ids, cluster_ids)
    |
    +---> Token embedding[token_id] ----+
    |                                   |
    +---> Value embedding[cluster_id] --+--> Addition --> rest of transformer
          (learned)
```

The key insight: **cluster_id is just another input feature**, like a learned positional encoding but for semantic context.

## Leg 1: Embedding the Training Set

### Goal

For every position in the training set, create an embedding of the context (last 32 tokens). Then cluster these embeddings to create our vocabulary of context patterns.

### Parameters

- **Context window**: 32 tokens (configurable)
- **Stride**: Every n tokens (n=1, or higher if that's too expensive)
- **Training data**: SlimPajama-627B (627B tokens)
- **Target clusters**: 100M clusters (to match Engram's scale)

### Calculations

**Number of embeddings to create:**
- If stride=1: ~627B embeddings (one per token position)
- If stride=8: ~78B embeddings

**Embedding dimensions:**
- Small embedding model: 384-dim (e.g., all-MiniLM-L6-v2) or 768-dim (e.g., BERT-base)

**Storage:**
- 78B embeddings × 384 dims × 4 bytes = 120 TB (fp32)
- 78B embeddings × 384 dims × 2 bytes = 60 TB (fp16)

This is too large to store directly. We need a **streaming** approach.

### Centroid Approach (No K-Means Needed)

Instead of expensive k-means clustering, we use **random window embeddings as centroids**:

1. Sample N random windows from random positions in random files
2. Weight sampling by text length for representative coverage
3. Embed them using embeddinggemma-300m
4. Use these embeddings directly as centroids for nearest-neighbor assignment

This works because centroids lie on the actual embedding manifold of real text.
See Phase 0 validation results below.

### Nearest Neighbor Search: FP16 Brute Force ✓ VALIDATED

We benchmarked various ANN approaches and found **FP16 brute force** is optimal:

| Method | Speed (100K centroids) | Recall |
|--------|------------------------|--------|
| Faiss IVF (nprobe=100) | 1,769 q/s | 100% but slow |
| Faiss HNSW | 11,470 q/s | 19% (poor for 768-dim) |
| FP32 brute force | 31K q/s | 100% |
| **FP16 brute force** | **287K q/s** | **99.7%** |
| FP8 brute force | similar | 96.9% |

**Matryoshka (truncated dimensions) doesn't help:**
| Dims | Recall |
|------|--------|
| 256 | 45.9% |
| 512 | 65.9% |
| 768 | 100% |

**Conclusion:** Use FP16 brute force with full 768-dim embeddings. Implemented in `topk.py`.

### Cost Estimate (Local GPU)

| Step | Compute | Storage | Time |
|------|---------|---------|------|
| Sample & embed 100K centroids | 1 GPU | 300 MB | ~15 min |
| Assign cluster IDs (per 1M windows) | 1 GPU | - | ~3.5 sec |
| **Scale to 100M centroids** | 1 GPU | 300 GB | ~3 days |

### Cost Estimate (Embedding APIs)

If using embedding APIs instead of local GPU:

**Tokens to embed:**
- 1B samples × 32 tokens per context = **32B tokens**

**API Pricing (as of Jan 2026):**

| Provider | Model | Price per 1M tokens | Cost for 32B tokens |
|----------|-------|---------------------|---------------------|
| OpenAI | text-embedding-3-small | $0.02 | **$640** |
| OpenAI | text-embedding-3-large | $0.13 | **$4,160** |
| Voyage AI | voyage-3-lite | $0.02 | **$640** |
| Voyage AI | voyage-3 | $0.06 | **$1,920** |
| Cohere | embed-v4.0 | $0.12 | **$3,840** |

**API approach is competitive with local GPU** (~$500-1000), especially with cheaper models.

**Additional considerations for API approach:**
- **Throughput**: At 10K RPM, 32B tokens / 32 tokens per request = 1B requests → 70 days at max rate
- **Batching**: Most APIs support batching (e.g., 2048 texts/request), which could reduce to ~1-2 days
- **Rate limits**: May need to negotiate higher limits or use multiple API keys
- **No GPU needed**: Can run from any machine with good network

**Recommendation**: For prototyping (1M-10M samples), APIs are convenient. For full 1B+ samples, local embedding model is more practical due to rate limits.

### Embedding Model: `google/embeddinggemma-300m` ✓ LOCKED

**Model specs:**
- 300M parameters, 768-dim output
- Uses sentence-transformers: `SentenceTransformer("google/embeddinggemma-300m")`
- Gated model, requires HuggingFace login + license acceptance

**Benchmark results (single GPU, 32-token sequences):**

| Model | Params | Dims | Throughput | 1B Embed Time |
|-------|--------|------|------------|---------------|
| all-MiniLM-L6-v2 | 22M | 384 | 5,733 texts/sec | 2.0 days |
| nomic-embed-text-v1.5 | 137M | 768 | 2,575 texts/sec | 4.5 days |
| **embeddinggemma-300m** | 300M | 768 | 838 texts/sec | **13.8 days** |
| **embeddinggemma-300m (compiled)** | 300M | 768 | **5,500 texts/sec** | **2.1 days** |

With `torch.compile(mode="max-autotune")` and fixed batch/seq shapes, throughput improves 6.5x.

**Multi-GPU scaling:**
- 8x GPUs: ~6 hours for 1B embeddings (with compiled model)
- Embarrassingly parallel - just shard the data

**Example usage:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

# For our use case (document/context embedding):
context = "The quick brown fox jumps over the lazy dog..."
embedding = model.encode_document(context)  # shape: (768,)
```

## Leg 2: Training the Value Embeddings

### Terminology

- **Keys**: Cluster centroids in the embedding model's output space (fixed after clustering)
- **Query**: Embedding of current context window (computed by embedding model)
- **Values**: Learned dense embeddings in a table that's part of the transformer model

The "sparsity" is in the **activation pattern** (only one cluster is looked up per position), not in the embeddings themselves. The value embeddings are dense vectors.

### Architecture

The embedding model runs **independently** from the main transformer. We precompute cluster IDs:

```
[Preprocessing - runs once on training data]

Context window (last 32 tokens)
    |
    v
Embedding model (frozen) --> query embedding
    |
    v
Nearest centroid lookup --> cluster_id
    |
    v
Store (token_position, cluster_id) mapping


[Training - cluster_id comes with each token]

Input: (tokens, cluster_ids)
    |
    v
Token embedding table[token_id] --> token_emb
Value embedding table[cluster_id] --> value_emb  (THIS IS LEARNED)
    |
    v
Combined: token_emb + value_emb (or concat, or other fusion)
    |
    v
Rest of transformer
```

### Key Insight

The cluster_id acts like an additional "context token ID" that captures the semantic context. The value embedding table is just another `nn.Embedding(num_clusters, hidden_dim)` that we train alongside the model.

### Training

1. **Preprocessing**: Run embedding model on all training data, compute cluster IDs, store as metadata
2. **Training loop**: Each batch includes `(input_ids, target_ids, cluster_ids)`
3. **Forward pass**: Look up both token embeddings and value embeddings, combine them
4. **Backward pass**: Gradients flow through value embeddings (cluster assignment is fixed/non-differentiable)

### Value Embedding Table Size

| Clusters | Hidden dim | Parameters | Memory (fp16) |
|----------|------------|------------|---------------|
| 1M | 768 | 768M | 1.5 GB |
| 10M | 768 | 7.68B | 15 GB |
| 100M | 768 | 76.8B | 150 GB |

100M clusters × 768 dims = **76.8B parameters** just in the value table. This matches Engram's scale (27B params in their memory module).

For a smaller experiment: 1M clusters is very manageable (1.5 GB).

### Open Questions

1. ~~**Which embedding model?**~~ → **RESOLVED**: `google/embeddinggemma-300m` (768d)

2. **Context window size?** 32 tokens seems reasonable, but should test 16, 64

3. **Number of clusters?** Start with 100K, then scale to 1M, 10M, 100M

4. **How to combine token_emb and value_emb?**
   - Addition (simplest, like Engram)
   - Concatenation + projection
   - Gating mechanism

5. **Where to inject?**
   - At input (layer 0) - simplest
   - At specific layers (like Engram)
   - Multiple layers with different value tables

6. **Initialization of value embeddings?**
   - Zero-init (start with no contribution)
   - Small random init
   - Initialize from cluster centroids (projected to hidden dim)

## Next Steps

1. [x] Set up embedding pipeline with embeddinggemma-300m
2. [x] Validate embedding similarity (Phase -1)
3. [x] Validate clustering approach with random window centroids (Phase 0)
4. [x] Validate nearest neighbor search → FP16 brute force (see `topk.py`)
5. [x] Generate and save centroids to disk (Phase 1) - 100K and 1M done
6. [x] Assign cluster IDs to training data (Phase 2) - `ClusterAssigner` done
7. [x] Integrate with transformer training loop (Phase 3) - model integration done
8. [ ] **Compare with baseline and Engram-style hashing (Phase 4)** ← NEXT

---

## Implementation Plan

### Phase 1: Generate Centroids ✓ DONE

Generate 100K random window embeddings and save to disk.

**Files:**
- `topk.py` - FP16/FP8 brute force top-k search ✓
- `embedder.py` - WindowEmbedder class with torch.compile ✓
- `sampling.py` - Random window sampling ✓
- `generate_centroids.py` - Script to generate centroids ✓
- `centroids.pt` - 100K centroid embeddings (to generate)

**Storage:**
- 100K × 768 × 4 bytes = 300 MB (FP32)
- 100K × 768 × 2 bytes = 150 MB (FP16)

**Tasks:**
- [x] 1.1: Script to sample 100K random windows from SlimPajama (`sampling.py`)
- [x] 1.2: Embed windows using embeddinggemma-300m with torch.compile (~2 min for 100K)
- [x] 1.3: Save centroids as FP16 tensor
  - `centroids_100k_w32_slimpajama.pt` (154 MB)
  - `centroids_1m_w32_slimpajama.pt` (1.5 GB)
- [x] 1.4: Validated via collision analysis and clustering tests

### Phase -1: Validate Embedding Similarity ✓ DONE

Tested with multiple random SlimPajama samples across 5 seeds. See `validate_embeddings.py`.

**Basic results (3 seeds):**
| Comparison | Cosine Similarity |
|------------|-------------------|
| Consecutive windows (shift=1 token) | **0.96** (std=0.03) |
| Non-consecutive windows (same text) | 0.90 (std=0.04) |
| Different texts | **0.13-0.17** |

**Similarity by distance (5 seeds, 245 samples):**
| Distance (tokens) | Mean Similarity | Interpretation |
|-------------------|-----------------|----------------|
| 1 | 0.96 | Adjacent windows (31/32 token overlap) |
| 8 | 0.83 | ~1 semantic unit apart |
| 16 | 0.71 | Half window overlap |
| 32 | 0.48 | No token overlap (full window shift) |
| 49 | 0.41 | Converges to document-level similarity |

**Key observations:**
- Similarity drops ~0.015 per token of distance
- At distance=32 (no overlap), similarity is ~0.48 - still higher than different texts (~0.16)
- Plateaus around 0.40-0.45 for large distances (topical/document-level similarity)
- Different documents: 0.11-0.31 depending on topic similarity

**Implications for clustering:**
- Windows shifted by just a few tokens will likely cluster together
- This is desirable: we want semantic similarity, not exact token matching
- May need stride > 1 when generating training cluster IDs to avoid redundancy

**Conclusion:** Embedding similarity correlates strongly with context similarity. ✓

### Phase 0: Validate Clustering/Indexing Approach ✓ DONE

**Approach: Random window embeddings as centroids**

Instead of expensive k-means clustering on the full dataset, we:
1. Sample N random windows from random positions in random files
2. Embed them using embeddinggemma-300m
3. Use these embeddings as centroids for nearest-neighbor assignment

Texts are sampled with weight proportional to their length for representative coverage.
Windows can be 1 to 32 tokens, allowing sampling from the beginning of texts.

See `test_random_centroids_consecutive.py`, `embedder.py`, and `sampling.py` for the implementation.

**Results with real window embeddings as centroids (5000 centroids, 200 texts):**

| Distance | Same Cluster % |
|----------|----------------|
| 1 | 80.6% |
| 2 | 72.6% |
| 4 | 61.2% |
| 8 | 42.5% |

**Cluster coverage (1000 centroids, 10k diverse windows):**

| Metric | Value |
|--------|-------|
| Centroids with 1+ matches | 96.5% |
| Mean matches per centroid | 10.0 |
| Median | 7.0 |
| Std | 10.3 |

**Key findings:**
- Adjacent windows share clusters ~81% of the time ✓
- Cluster usage is well-distributed (96.5% of centroids used) ✓
- Real window embeddings as centroids work well - no k-means needed

### Collision Analysis (100K centroids)

Tested 1000 random windows → 977 unique centroids (21 collisions).
Collisions show semantically similar content clustering together:

**Example collisions:**

1. **Academic/event announcements** (centroid 79220):
   - "was held on 30th April, where our students and faculty gathered..."
   - "Our BA Practical Filmmaking students were honoured to receive a great guest lecture"
   - "Rent-to-Own Personal Property, Is it Taxable or Exempt? This panel will make..."

2. **Philosophical/reflective prose** (centroid 70193):
   - "be the cure for the other, right? It's actually a little more complex..."
   - "are not always apparent on a first hearing. Like Easter eggs hidden in a field..."
   - "self-evident. When you delude yourself into thinking that you see something..."

3. **CSS/code patterns** (centroid 82016):
   - `btn.active, .btn:active:focus, .btn:active:hover...`
   - `btn-neutral:disabled:active, .btn-neutral:disabled.active...`

4. **Django model fields** (centroid 10230):
   - `is_removed': ('django.db.models.fields.BooleanField'...`
   - `description = _("The base GIS field.") empty_strings_allowed = False...`

5. **COVID remote learning** (centroid 10101) - near-duplicate:
   - "to flatten the curve of infection. Though some schools, like Friends' Central..."
   - "towards remote learning, driven by the need to embrace social distancing..."

See `find_collisions.py` for analysis script.

### Phase 2: Centroid Generation & Cluster Assignment

- [x] 2.1: Library to sample random windows from SlimPajama (length-weighted) ✓ DONE (`sampling.py`)
- [x] 2.2: Script to embed centroid windows using embeddinggemma-300m ✓ DONE (`generate_centroids.py`)
- [x] 2.3: Library to index from centroid embeddings including topk ✓ DONE (`topk.py`)
- [x] 2.4: WindowEmbedder class with torch.compile optimization ✓ DONE (`embedder.py`)
- [x] 2.5: Generate and save 100K+ centroids to disk ✓ DONE (100K and 1M saved)
- [x] 2.6: ClusterAssigner class for assigning training positions to nearest centroids ✓ DONE (`cluster_assigner.py`)

### ClusterAssigner Implementation Details

**File:** `cluster_assigner.py`

**Causality guarantee:** For position p, the cluster_id is computed from tokens[max(0, p-32):p],
which excludes the token at position p. This ensures no future information leakage for autoregressive models.

**Usage:**
```python
from cluster_assigner import ClusterAssigner

assigner = ClusterAssigner("centroids_1m_w32_slimpajama.pt", window_size=32)
cluster_ids = assigner.assign_single(token_ids, stride=8)  # Returns (seq_len // stride,) tensor
```

**Performance (1M centroids, compiled embedder):**
| Operation | Time |
|-----------|------|
| Load assigner (first time) | ~3s |
| First batch (autotuning) | ~12s |
| Subsequent batches | ~0.1s per text |

**Integration approaches:**
1. **JIT in training loop**: Compute cluster_ids on-the-fly during training
2. **Precompute**: Run `assign_batch` on all training data, save sidecar files

### Phase 3: Model Integration ✓ DONE

- [x] 3.1: Add `nn.Embedding(num_clusters, hidden_dim)` to model (`transformer.py`)
- [x] 3.2: Add `cluster_ids` field to `LMData` (`language_model_basics.py`)
- [x] 3.3: Modify forward pass to add value embeddings to residual stream
- [x] 3.4: Add JIT cluster assignment in training loop (`language_model_training.py`)
- [x] 3.5: Test integration end-to-end

**Config options added to TransformerConfig:**
- `chunk_embeddings: bool = False` - Enable chunk embeddings
- `chunk_num_clusters: int = 1_000_000` - Number of clusters
- `chunk_window_size: int = 32` - Context window size
- `chunk_init_scale: float = 0.0` - Initial scale (0 = zero-init)

**Usage:**
```python
# Enable in model
config = TransformerConfig(chunk_embeddings=True, chunk_num_clusters=1_000_000)
model = TransformerModel(vocab_size, config)

# Training with JIT cluster assignment
from experiments.chunk_embeddings.cluster_assigner import ClusterAssigner
assigner = ClusterAssigner("centroids_1m_w32_slimpajama.pt")
training_state = LanguageModelTrainingState(model, train_config, cluster_assigner=assigner)
```

### Phase 4: Evaluation

- [ ] 4.1: Baseline: train model without chunk embeddings
- [ ] 4.2: Experiment: train model with chunk embeddings
- [ ] 4.3: Compare loss curves, downstream metrics
- [ ] 4.4: Ablations: cluster count, context window size, injection point

---

## Future Work

### Top-K Retrieval Instead of Top-1

Currently we assign each position to its single nearest centroid (top-1). An extension is to retrieve **top-K centroids** (e.g., K=16) and combine their value embeddings:

**Options for combining top-K:**
1. **Weighted sum**: `value_emb = sum(similarity[i] * value_table[centroid[i]] for i in range(K))`
2. **Attention**: Use similarities as attention weights over value embeddings
3. **Concatenate + project**: `value_emb = Linear(concat(value_table[centroid[0:K]]))`

**Potential benefits:**
- Smoother interpolation between similar contexts
- More robust to centroid boundary effects
- Richer information per position

**Considerations:**
- K× more memory bandwidth for value table lookups
- Need to store K cluster IDs per position (or recompute at training time)
- May need to tune K vs. number of centroids trade-off
