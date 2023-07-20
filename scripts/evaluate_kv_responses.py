#!/usr/bin/env python3
"""Given a data file with LM predictions for kv retrieval, evaluate the predictions.
"""
import argparse
import json
import logging
import statistics
import sys
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

logger = logging.getLogger(__name__)


def main(input_path, output_path):
    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)

    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        model_answer = example["model_answer"]

        accuracy = 1.0 if example["value"].lower() in model_answer.lower() else 0.0

        all_example_metrics.append(({"accuracy": accuracy}, example))

    # Average metrics across examples
    for metric_name in ["accuracy"]:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name}: {average_metric_value}")

    if output_path:
        with xopen(output_path, "w") as f:
            for (example_metrics, example) in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                f.write(json.dumps(example_with_metrics) + "\n")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with model predictions and answers.", required=True)
    parser.add_argument(
        "--output-path",
        help="Path to write data with model predictions, answers, and scores.",
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(args.input_path, args.output_path)
    logger.info("finished running %s", sys.argv[0])
