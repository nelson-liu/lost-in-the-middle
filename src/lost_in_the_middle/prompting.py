#!/usr/bin/env python3
import pathlib
from copy import deepcopy
from typing import List, Optional, Tuple, Type, TypeVar

from pydantic.dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()

T = TypeVar("T")


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def get_qa_prompt(
    question: str, documents: List[Document], mention_random_ordering: bool, query_aware_contextualization: bool
):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    if mention_random_ordering and query_aware_contextualization:
        raise ValueError("Mentioning random ordering cannot be currently used with query aware contextualization")

    if mention_random_ordering:
        prompt_filename = "qa_ordered_randomly.prompt"
    elif query_aware_contextualization:
        prompt_filename = "qa_with_query_aware_contextualization.prompt"
    else:
        prompt_filename = "qa.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_closedbook_qa_prompt(question: str):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    with open(PROMPTS_ROOT / "closedbook_qa.prompt") as f:
        prompt_template = f.read().rstrip("\n")

    return prompt_template.format(question=question)


def get_kv_retrieval_prompt(
    data: List[Tuple[str, str]],
    key: str,
    query_aware_contextualization: bool = False,
):
    if not data:
        raise ValueError(f"Provided `data` must be truthy, got: {data}")
    if not key:
        raise ValueError(f"Provided `key` must be truthy, got: {key}")
    if key not in [x[0] for x in data]:
        raise ValueError(f"Did not find provided `key` {key} in data {data}")
    if len(data) != len(set([x[0] for x in data])):
        raise ValueError(f"`data` has duplicate keys: {data}")
    if len(data) < 2:
        raise ValueError(f"Must have at least 2 items in data: {data}")

    if query_aware_contextualization:
        with open(PROMPTS_ROOT / "kv_retrieval_with_query_aware_contextualization.prompt") as f:
            prompt_template = f.read().rstrip("\n")
    else:
        with open(PROMPTS_ROOT / "kv_retrieval.prompt") as f:
            prompt_template = f.read().rstrip("\n")

    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        start_character = "{" if index == 0 else " "
        data_string = f'"{record[0]}": "{record[1]}"'
        end_character = ",\n" if index != len(data) - 1 else "}"
        formatted_kv_records += start_character + data_string + end_character

    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)
