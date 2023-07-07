# Lost in the Middle: How Language Models Use Long Contexts

This repository contains accompanying material for Lost in the Middle: How Language Models Use Long Contexts.

## Multi-Document Question Answering Data

[`qa_data/`](./qa_data/) contains multi-document question answering data for the
oracle setting (1 input document, which is exactly the passage the answers the
question) and 10-, 20-, and 30-document settings (where 1 input passage answers
the question, and the other passages do not contain an NQ-annotated answer).

Each line of this gzipped file is in the following format:

``` sh
{
  "question": "who got the first nobel prize in physics",
  "answers": [
    "Wilhelm Conrad Röntgen"
  ],
  "ctxs": [
    ...
    {
      "id": <string id, e.g., "71445">,
      "title": <string title of the wikipedia article that this passage comes from>,
      "text": <string content of the passage>,
      "score": <string relevance score, e.g. "1.0510446">,
      "hasanswer": <boolean, whether any of the values in the `answers` key appears in the text>,
      "original_retrieval_index": <int indicating the original retrieval index. for example, a value of 0 indicates that this was the top retrieved document>,
      "isgold": <boolean, true or false indicating if this chunk is the gold answer from NaturalQuestions>
    },
    ...
  ],
  "nq_annotated_gold": {
    "title": <string title of the wikipedia article containing the answer, as annotated in NaturalQuestions>,
    "long_answer": "<string content of the paragraph element containing the answer, as annotated in NaturalQuestions>",
    "chunked_long_answer": "<string content of the paragraph element containing the answer, randomly chunked to approximately 100 words>",
    "short_answers": [
      <string short answers, as annootated in NaturalQuestions>
    ]
  }
}
```

### Generating new multi-document QA data.

1. First, download Contriever retrieval results for each of the queries:

``` sh
wget https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz
```

2. Then, to generate examples with 20 total documents with the relevant documents at positions 0, 4, 9, 14, and 19, run:

``` sh
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/make_qa_data_from_retrieval_results.py \
        --input-path nq-open-contriever-msmarco-retrieved-documents.jsonl.gz \
        --num-total-documents 20 \
        --gold-index ${gold_index} \
        --output-path qa_data/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz
done
```

## Key-Value Retrieval Data

[`kv_retrieval_data/`](./kv_retrieval_data/) contains multi-document question answering data for the
oracle setting (1 input document, which is exactly the passage the answers the
question) and 10-, 20-, and 30-document settings (where 1 input passage answers
the question, and the other passages do not contain an NQ-annotated answer).

Each line of this gzipped file is in the following format:

``` sh
{
  "ordered_kv_records": [
    ...
    [
      "adde4211-888b-48e3-9dbe-66c66551b05f",
      "8bc3e90d-e044-4923-9a5d-7bf48917ed39"
    ],
    [
      "2a8d601d-1d69-4e64-9f90-8ad825a74195",
      "bb3ba2a5-7de8-434b-a86e-a88bb9fa7289"
    ],
    [
      "006b46ef-03fd-4380-938c-49cb03754370",
      "9faeacbe-1d0e-40da-a5db-df598a104880"
    ],
    ...
  ],
  "key": "2a8d601d-1d69-4e64-9f90-8ad825a74195",
  "value": "bb3ba2a5-7de8-434b-a86e-a88bb9fa7289"
}
```

The `ordered_kv_records` is a list of `[key, value]` pairs. The `key` specifies
a particular key to retrieve from `ordered_kv_records`, and the `value` lists
its expected associated value.

### Manipulating the gold index in key-value retrieval data

To manipulate the gold index in key-value retrieval data, modify the
`ordered_kv_records`. For example, for `gold_index = 0` (put the key-value
pair to retrieve at the very start of the input context).

``` python
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

gold_index = 0
gold_index_kv_examples = []

with xopen(input_path) as fin:
    for line in fin:
        input_example = json.loads(line)

        # Get the prediction for the input example
        ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
        key = input_example["key"]
        value = input_example["value"]
        if gold_index is not None:
            original_kv_index = ordered_kv_records.index([key, value])
            # Remove the kv from its original index
            original_kv = ordered_kv_records.pop(original_kv_index)
            ordered_kv_records.insert(gold_index, original_kv)

        gold_index_kv_examples.append({
            "key": key,
            "value": value,
            "ordered_kv_records": ordered_kv_records
        })
```

### Generating new key-value retrieval data

To generate new key-value retrieval data, use:

``` sh
python -u ./scripts/make_kv_retrieval_data.py \
    --num-keys 300 \
    --num-examples 500 \
    --output-path kv-retrieval_data/kv-retrieval-300_keys.jsonl.gz
```

## Prompting

Code for converting the examples into string prompts is in
[`prompts/`](./prompts). After installing `pip` requirements, you can run tests
with:

``` sh
$ py.test prompts/test_prompting.py
========================================= test session starts =========================================
platform linux -- Python 3.10.12, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/nfliu/git/lost-in-the-middle
collected 7 items

prompts/test_prompting.py .......                                                               [100%]

========================================== 7 passed in 0.08s ==========================================
```

## References

Please consider citing our work if you found this code or our paper beneficial to your research.

```
@misc{liu-etal:2023:arxiv,
  author    = {Nelson F. Liu  and  Kevin Lin  and  John Hewitt  and Ashwin Paranjape  and Michele Bevilacqua  and  Fabio Petroni  and  Percy Liang},
  title     = {Lost in the Middle: How Language Models Use Long Contexts},
  note      = {},
  year      = {2023}
}

```
