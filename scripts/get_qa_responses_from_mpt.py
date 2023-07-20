#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run MPT to get responses.

Currently supports `mosaicml/mpt-30b-instruct` and `mosaicml/mpt-30b`.

The retrieval results are used in the exact order that they're given.
"""
import argparse
import dataclasses
import json
import logging
import math
import pathlib
import random
import sys
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xopen import xopen

from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    batch_size,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    num_gpus,
    max_memory_per_gpu,
    max_new_tokens,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            if "instruct" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an instruct model, applying instruct formatting")
                    did_format_warn = True
                prompt = format_instruct_prompt(prompt)
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    # Get responses for all of the prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.attn_config["attn_impl"] = "triton"
    config.max_seq_len = 16384
    # (input + output) tokens can now be up to 16384
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    logger.info("Loading model")
    if num_gpus > 1:
        extra_kwargs = {
            "device_map": "auto",
            "max_memory": {i: f"{str(max_memory_per_gpu)}GiB" for i in range(num_gpus)},
        }
    else:
        extra_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **extra_kwargs,
    )
    # Move model to GPU if we're using single-gpu
    if num_gpus == 1:
        model = model.to(device)

    print(model)

    do_sample = temperature > 0.0

    responses = []
    with torch.autocast(device, dtype=torch.bfloat16):
        for batched_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts) / batch_size)):
            inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                use_cache=True,
                eos_token_id=0,
                pad_token_id=0,
                return_dict_in_generate=False,
            )
            for i, generated_sequence in enumerate(outputs):
                input_ids = inputs[i].ids
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    )
                new_text = text[prompt_length:]
                responses.append(new_text)

    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def format_instruct_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT_FOR_GENERATION = "{intro}\n{instruction_key}\n{instruction}\n{response_key}\n".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction=instruction,
        response_key=RESPONSE_KEY,
    )
    return PROMPT_FOR_GENERATION


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
        choices=["mosaicml/mpt-30b-instruct", "mosaicml/mpt-30b"],
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--batch-size", help="Batch size use in generation", type=int, default=8)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--max-memory-per-gpu",
        help="Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80",
        type=int,
    )
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.batch_size,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.num_gpus,
        args.max_memory_per_gpu,
        args.max_new_tokens,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
