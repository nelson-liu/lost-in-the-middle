# Multi-Document Question Answering

Note: all of these experiments were run on one or more A100 GPUs with 80GB of
VRAM. You may need to modify commands to fit your own computing environment
(e.g., changing the batch size, the max memory per GPU, the number of GPUs, etc)

## mpt-30b-instruct

To run `mpt-30b` and `mpt-30b-instruct` on multi-document question answering,
use [`./scripts/get_qa_responses_from_mpt.py`](./scripts/get_qa_responses_from_mpt.py).
Below are commands for running `mpt-30b-instruct` on different multi-document QA
settings.

### mpt-30b-instruct on oracle

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_mpt.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 1 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --model mosaicml/mpt-30b-instruct \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-mpt-30b-instruct-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.816572504708098
```

### mpt-30b-instruct on closedbook

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_mpt.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 1 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --closedbook \
    --model mosaicml/mpt-30b-instruct \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-closedbook-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-mpt-30b-instruct-closedbook-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-mpt-30b-instruct-closedbook-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.3167608286252354
```

### mpt-30b-instruct on 20-document setting

Getting predictions:

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/get_qa_responses_from_mpt.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --num-gpus 1 \
        --max-new-tokens 100 \
        --batch-size 1 \
        --max-memory-per-gpu 80 \
        --num-gpus 1 \
        --model mosaicml/mpt-30b-instruct \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-mpt-30b-instruct-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.536723163841808

Gold Index 4:
best_subspan_em: 0.5175141242937853

Gold Index 9:
best_subspan_em: 0.5216572504708098

Gold Index 14:
best_subspan_em: 0.5265536723163842

Gold Index 19:
best_subspan_em: 0.5623352165725047
```

## longchat-13b-16k

To run `longchat-13b-16k` on multi-document question answering,
use [`./scripts/get_qa_responses_from_longchat.py`](./scripts/get_qa_responses_from_longchat.py).
Below are commands for running `longchat-13b-16k` on different multi-document QA
settings.

### longchat-13b-16k on oracle

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_longchat.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 8 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --model lmsys/longchat-13b-16k \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-longchat-13b-16k-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.8263653483992467
```

### longchat-13b-16k on closedbook

Getting predictions:

```
python -u ./scripts/get_qa_responses_from_longchat.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 8 \
    --max-memory-per-gpu 80 \
    --num-gpus 1 \
    --closedbook \
    --model lmsys/longchat-13b-16k \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions.jsonl.gz
```

Evaluating: 

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions-scored.jsonl.gz
```

You should get something approximately around:

```
best_subspan_em: 0.34990583804143127
```

### longchat-13b-16k on 20-document setting

Getting predictions:

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/get_qa_responses_from_longchat.py \
        --input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
        --num-gpus 1 \
        --max-new-tokens 100 \
        --batch-size 1 \
        --max-memory-per-gpu 80 \
        --num-gpus 1 \
        --model lmsys/longchat-13b-16k \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz \
        --output-path qa_predictions/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-longchat-13b-16k-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.6858757062146893

Gold Index 4:
best_subspan_em: 0.5740112994350283

Gold Index 9:
best_subspan_em: 0.5532956685499059

Gold Index 14:
best_subspan_em: 0.5250470809792843

Gold Index 19:
best_subspan_em: 0.5502824858757062
```

# Key-Value Retrieval

Note: all of these experiments were run on one or more A100 GPUs with 80GB of
VRAM. You may need to modify commands to fit your own computing environment
(e.g., changing the batch size, the max memory per GPU, the number of GPUs, etc)

## mpt-30b-instruct

To run `mpt-30b` and `mpt-30b-instruct` on key-value retrieval, use
[`./scripts/get_kv_responses_from_mpt.py`](./scripts/get_kv_responses_from_mpt.py).
Below are commands for running `mpt-30b-instruct` on different KV retrieval
settings.

### mpt-30b-instruct with 140 total key-value pairs

Getting predictions:

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/get_kv_responses_from_mpt.py \
        --input-path kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz \
        --batch-size 1 \
        --gold-index ${gold_index} \
        --model mosaicml/mpt-30b-instruct \
        --max-memory-per-gpu 80 \
        --num-gpus 1 \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/evaluate_kv_responses.py \
        --input-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-mpt-30b-instruct-predictions.jsonl.gz \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-mpt-30b-instruct-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 1.0

Gold Index 34:
best_subspan_em: 0.936

Gold Index 69:
best_subspan_em: 0.886

Gold Index 104:
best_subspan_em: 0.804

Gold Index 139:
best_subspan_em: 0.962
```

## longchat-13b-16k

To run `longchat-13b-16k` on key-value retrieval, use
[`./scripts/get_kv_responses_from_longchat.py`](./scripts/get_kv_responses_from_mpt.py).
Below are commands for running `longchat-13b-16k` on different KV retrieval
settings.

### longchat-13b-16k with 140 total key-value pairs

Getting predictions:

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/get_kv_responses_from_longchat.py \
        --input-path kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz \
        --batch-size 1 \
        --gold-index ${gold_index} \
        --max-memory-per-gpu 80 \
        --num-gpus 2 \
        --model lmsys/longchat-13b-16k \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz
done
```

Evaluating: 

```
for gold_index in 0 34 69 104 139; do
    python -u ./scripts/evaluate_kv_responses.py \
        --input-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-longchat-13b-16k-predictions.jsonl.gz \
        --output-path kv_predictions/kv-retrieval-140_keys_gold_at_${gold_index}-longchat-13b-16k-predictions-scored.jsonl.gz
done
```

You should get something approximately around:

```
Gold Index 0:
best_subspan_em: 0.37

Gold Index 34:
best_subspan_em: 0.382

Gold Index 69:
best_subspan_em: 0.354

Gold Index 104:
best_subspan_em: 0.656

Gold Index 139:
best_subspan_em: 0.886
```
