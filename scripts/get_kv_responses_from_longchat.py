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
from fastchat.model import get_conversation_template, load_model
from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.prompting import get_kv_retrieval_prompt

logger = logging.getLogger(__name__)
random.seed(0)


# Copied from https://github.com/DachengLi1/LongChat/blob/43d71f03d7711a2ab3b78ee8d1e38b65bb7fd22f/longeval/utils.py
def maybe_monkey_patch(model_name: str, longchat_flash_attn: bool, longchat_ratio: int):
    if "longchat" in model_name:
        from longchat.train.monkey_patch.llama_condense_monkey_patch import (
            replace_llama_with_condense,
        )

        replace_llama_with_condense(longchat_ratio)

        if longchat_flash_attn:
            from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import (
                replace_llama_attn_with_flash_attn,
            )

            replace_llama_attn_with_flash_attn()
    import transformers  # noqa: F401


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
    longchat_flash_attn,
    longchat_ratio,
    max_new_tokens,
    output_path,
):
    if longchat_ratio != 8:
        raise ValueError("--longchat-ratio=8 is the only value currently supported.")

    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    maybe_monkey_patch(model_name=model_name, longchat_flash_attn=longchat_flash_attn, longchat_ratio=longchat_ratio)
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
            # Remove the kv from its original index
            original_kv = ordered_kv_records.pop(original_kv_index)
            ordered_kv_records.insert(gold_index, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
            )

            if "chat" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
                kv_prompt = format_chat_prompt(kv_prompt)
            prompts.append(kv_prompt)
            examples.append(deepcopy(input_example))
            all_model_ordered_kv_records.append(ordered_kv_records)

    # Get responses for all of the prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    model, tokenizer = load_model(
        model_name,
        device="cuda",
        num_gpus=num_gpus,
        max_gpu_memory=f"{max_memory_per_gpu}GiB",
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    tokenizer.padding_side = "left"
    print(model)

    do_sample = temperature > 0.0

    responses = []
    for batched_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts) / batch_size)):
        inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            # Disable use_cache if using longchat models with flash attention
            use_cache=not ("longchat" in model_name and longchat_flash_attn),
        )
        for i, generated_sequence in enumerate(outputs):
            input_ids = inputs["input_ids"][i]
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


def format_chat_prompt(input):
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model", help="Model to use in generating responses", required=True, choices=["lmsys/longchat-13b-16k"]
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--batch-size", help="Batch size use in generation", type=int, default=8)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument("--gold-index", help="Move the key to retrieve to this index", type=int, required=True)
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int)
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument(
        "--longchat-flash-attn",
        action="store_true",
        help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.",
    )
    parser.add_argument(
        "--longchat-ratio",
        type=int,
        default=8,
        help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        help="Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80",
        type=int,
    )
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
        args.gold_index,
        args.num_gpus,
        args.max_memory_per_gpu,
        args.query_aware_contextualization,
        args.longchat_flash_attn,
        args.longchat_ratio,
        args.max_new_tokens,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
