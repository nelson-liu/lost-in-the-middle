#!/usr/bin/env python3
"""Given a data file with KV records, get LM retrieval results.

The KV records are used in the exact order that they're given.
"""
import argparse
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

from lost_in_the_middle.prompting import get_kv_retrieval_prompt

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    batch_size,
    gold_index,
    num_gpus,
    max_memory_per_gpu,
    query_aware_contextualization,
    max_new_tokens,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_ordered_kv_records = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
            key = input_example["key"]
            value = input_example["value"]
            original_kv_index = ordered_kv_records.index([key, value])
            # Remove the kv to retrieve from its original index
            original_kv = ordered_kv_records.pop(original_kv_index)
            # Insert it at the specified gold index
            ordered_kv_records.insert(gold_index, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
            )

            if "instruct" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an instruct model, applying instruct formatting")
                    did_format_warn = True
                kv_prompt = format_instruct_prompt(kv_prompt)
            prompts.append(kv_prompt)
            examples.append(deepcopy(input_example))
            all_model_ordered_kv_records.append(ordered_kv_records)

    # Get responses for all of the prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.attn_config["attn_impl"] = "triton"
    config.max_seq_len = 16384  # (input + output) tokens can now be up to 16384

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
        for example, ordered_kv_records, prompt, response in zip(
            examples, all_model_ordered_kv_records, prompts, responses
        ):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_ordered_kv_records"] = ordered_kv_records
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
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument("--gold-index", help="Move the key to retrieve to this index", type=int, required=True)
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--query-aware-contextualization", action="store_true", help="Use query-aware contextualization"
    )
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        help="Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80",
        type=int,
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.batch_size,
        args.gold_index,
        args.num_gpus,
        args.max_memory_per_gpu,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
