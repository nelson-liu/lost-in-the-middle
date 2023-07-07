#!/usr/bin/env python3
"""Given a data file with retrieval results, create multi-document QA data
where exactly 1 passage contains the answer, and N-1 passages are distractors
that do not contain the NQ-annotated answer.

This data is meant to simulate a realistic retrieval setting where we retrieve N
documents and 1 is relevant (and answers question), while the other N - 1 do not.

Running:

```
python -u ./scripts/make_qa_data_from_retrieval_results.py \
    --input-path nq-open-contriever-msmarco-retrieved-documents.jsonl.gz \
    --num-total-documents 30 \
    --gold-index 0 \
    --output-path qa_data/nq-open-30_total_documents_gold_at_0.jsonl.gz
```

"""
import argparse
import json
import logging
import sys
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

logger = logging.getLogger(__name__)


def main(input_path, num_total_documents, gold_index, output_path):

    if num_total_documents < 2:
        raise ValueError(f"`num_total_documents` must be at least 2, got {num_total_documents}")
    if gold_index < 0:
        raise ValueError("`gold_index` must be at least 0")
    if gold_index >= num_total_documents:
        raise ValueError(f"`gold_index` must be less than `num_total_documents` ({num_total_documents})")

    # Validate that we have at least num_total_documents for every example
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            qa_retrieval_result = json.loads(line)
            example_num_documents = len([doc for doc in qa_retrieval_result["ctxs"] if doc["hasanswer"] is False])
            if num_total_documents > example_num_documents:
                raise ValueError(
                    f"Requested `num_total_documents` {num_total_documents}, but found an input"
                    f"example with only {example_num_documents} documents that don't contain the answer."
                )

    num_output_examples = 0
    with xopen(input_path) as fin, xopen(output_path, "w") as fout:
        for line in tqdm(fin):
            qa_retrieval_result = json.loads(line)
            # Get documents that don't contain the answer
            valid_distractors_with_retrieval_indices = [
                (idx, doc) for idx, doc in enumerate(qa_retrieval_result["ctxs"]) if doc["hasanswer"] is False
            ]
            # Take the top `num_total_documents - 1` distractors
            distractor_docs_with_retrieval_indices = deepcopy(
                valid_distractors_with_retrieval_indices[: num_total_documents - 1]
            )
            for original_retrieval_index, distractor_doc in distractor_docs_with_retrieval_indices:
                distractor_doc["original_retrieval_index"] = original_retrieval_index
                distractor_doc["isgold"] = False
            distractor_docs = [x[1] for x in distractor_docs_with_retrieval_indices]

            content_selection_example = deepcopy(qa_retrieval_result)
            gold_chunk = {
                "title": qa_retrieval_result["nq_annotated_gold"]["title"],
                "text": qa_retrieval_result["nq_annotated_gold"]["chunked_long_answer"],
                "hasanswer": True,
                "isgold": True,
            }
            ctxs = distractor_docs
            # Insert the gold chunk at thet specific index
            ctxs.insert(gold_index, gold_chunk)

            content_selection_example["ctxs"] = ctxs
            fout.write(json.dumps(content_selection_example) + "\n")
            num_output_examples += 1
    logger.info(f"Wrote {num_output_examples} output examples")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        help=("Path to (possibly gzipped) file with contriever retrieval results, annotated for gold passages."),
        required=True,
    )
    parser.add_argument(
        "--num-total-documents",
        help=(
            "# of total documents to use in content selection data. 1 will be gold, and the others will be "
            "taken from retrieval results. Must be at least 2."
        ),
        type=int,
        required=True,
    )
    parser.add_argument(
        "--gold-index",
        help=("Index to place gold documents at. Must be 0 or greater and " "`num-total-documents - 1` or smaller"),
        type=int,
        required=True,
    )
    parser.add_argument("--output-path", help="Path to write output data files", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(args.input_path, args.num_total_documents, args.gold_index, args.output_path)
    logger.info("finished running %s", sys.argv[0])
