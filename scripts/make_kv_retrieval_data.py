#!/usr/bin/env python3
"""Create a data file for key-value retrieval.

Running:

```
python -u ./scripts/make_kv_retrieval_data.py \
    --num-keys 300 \
    --num-examples 500 \
    --output-path data/kv-retrieval/kv-retrieval-300_keys.jsonl
```

"""
import argparse
import json
import logging
import random
import sys
import uuid

from tqdm import tqdm
from xopen import xopen

logger = logging.getLogger(__name__)


def main(num_keys, num_examples, output_path):

    if num_keys < 2:
        raise ValueError(f"`num_keys` must be at least 2, got {num_keys}")
    if num_examples < 1:
        raise ValueError(f"`num_examples` must be at least 1, got {num_examples}")

    num_output_examples = 0
    with xopen(output_path, "w") as fout:
        for _ in tqdm(range(num_examples)):
            kv_records = {}
            while len(kv_records) != num_keys:
                kv_records[str(uuid.uuid4())] = str(uuid.uuid4())
            ordered_kv_records = list(kv_records.items())
            gold_record = random.choice(ordered_kv_records)
            key = gold_record[0]
            value = gold_record[1]
            example = {"ordered_kv_records": ordered_kv_records, "key": key, "value": value}
            fout.write(json.dumps(example) + "\n")
            num_output_examples += 1
    logger.info(f"Wrote {num_output_examples} output examples")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-keys",
        help="# of keys to use in the KV retrieval data",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num-examples",
        help="# of examples to generate",
        type=int,
        required=True,
    )
    parser.add_argument("--output-path", help="Path to write output data files", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(args.num_keys, args.num_examples, args.output_path)
    logger.info("finished running %s", sys.argv[0])
