#!/usr/bin/env python3
import pytest

from .prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_kv_retrieval_prompt,
    get_qa_prompt,
)


def test_document_from_dict():
    document_data = {
        "id": "9459602",
        "title": "Science and technology in Germany",
        "text": "some text here",
        "score": "0.97236913",
        "hasanswer": True,
        "original_retrieval_index": 65,
    }
    document = Document.from_dict(document_data)
    assert document.id == document_data["id"]
    assert document.title == document_data["title"]
    assert document.text == document_data["text"]
    assert document.score == float(document_data["score"])
    assert document.hasanswer == document_data["hasanswer"]
    assert document.original_retrieval_index == document_data["original_retrieval_index"]


def test_get_qa_prompt():
    question = "test question"
    documents = [
        Document(
            id="1", title="first doc", text="first doc text", score=0.95, hasanswer=False, original_retrieval_index=0
        ),
        Document(
            id="2", title="second doc", text="second doc text", score=0.90, hasanswer=True, original_retrieval_index=1
        ),
    ]
    qa_prompt = get_qa_prompt(
        question=question, documents=documents, mention_random_ordering=False, query_aware_contextualization=False
    )
    gold_instruction = (
        "Write a high-quality answer for the given question using only "
        "the provided search results (some of which might be irrelevant)."
    )
    gold_formatted_doc_1 = "Document [1](Title: first doc) first doc text"
    gold_formatted_doc_2 = "Document [2](Title: second doc) second doc text"
    gold_prompt = (
        f"{gold_instruction}\n\n{gold_formatted_doc_1}\n{gold_formatted_doc_2}\n\nQuestion: {question}\nAnswer:"
    )
    assert qa_prompt == gold_prompt


def test_get_qa_prompt_mention_random_ordering():
    question = "test question"
    documents = [
        Document(
            id="1", title="first doc", text="first doc text", score=0.95, hasanswer=False, original_retrieval_index=0
        ),
        Document(
            id="2", title="second doc", text="second doc text", score=0.90, hasanswer=True, original_retrieval_index=1
        ),
    ]
    qa_prompt = get_qa_prompt(
        question=question, documents=documents, mention_random_ordering=True, query_aware_contextualization=False
    )
    gold_instruction = (
        "Write a high-quality answer for the given question using only "
        "the provided search results (some of which might be irrelevant). "
        "The search results are ordered randomly."
    )
    gold_formatted_doc_1 = "Document [1](Title: first doc) first doc text"
    gold_formatted_doc_2 = "Document [2](Title: second doc) second doc text"
    gold_prompt = (
        f"{gold_instruction}\n\n{gold_formatted_doc_1}\n{gold_formatted_doc_2}\n\nQuestion: {question}\nAnswer:"
    )
    assert qa_prompt == gold_prompt


def test_get_qa_prompt_query_aware_contextualization():
    question = "test question"
    documents = [
        Document(
            id="1", title="first doc", text="first doc text", score=0.95, hasanswer=False, original_retrieval_index=0
        ),
        Document(
            id="2", title="second doc", text="second doc text", score=0.90, hasanswer=True, original_retrieval_index=1
        ),
    ]
    qa_prompt = get_qa_prompt(
        question=question, documents=documents, mention_random_ordering=False, query_aware_contextualization=True
    )
    gold_instruction = (
        "Write a high-quality answer for the given question using only "
        "the provided search results (some of which might be irrelevant)."
    )
    gold_formatted_doc_1 = "Document [1](Title: first doc) first doc text"
    gold_formatted_doc_2 = "Document [2](Title: second doc) second doc text"
    gold_prompt = (
        f"{gold_instruction}\n\nQuestion: {question}\n\n{gold_formatted_doc_1}\n"
        f"{gold_formatted_doc_2}\n\nQuestion: {question}\nAnswer:"
    )
    assert qa_prompt == gold_prompt

    with pytest.raises(ValueError):
        for model_type in "unsupported":
            get_qa_prompt(
                question=question, documents=documents, mention_random_ordering=True, query_aware_contextualization=True
            )


def test_get_closedbook_qa_prompt():
    question = "test question"
    closedbook_qa_prompt = get_closedbook_qa_prompt(question=question)
    gold_prompt = f"Question: {question}\nAnswer:"
    assert closedbook_qa_prompt == gold_prompt


def test_get_kv_retrieval_prompt():
    data = [("test key1", "test value1"), ("test key2", "test value2"), ("test key3", "test value3")]
    key = "test key2"

    prompt = get_kv_retrieval_prompt(data=data, key=key)
    gold = "\n".join(
        [
            "Extract the value corresponding to the specified key in the JSON object below.",
            "",
            "JSON data:",
            "{\"test key1\": \"test value1\",",
            " \"test key2\": \"test value2\",",
            " \"test key3\": \"test value3\"}",
            "",
            "Key: \"test key2\"",
            "Corresponding value:",
        ]
    )
    assert prompt == gold


def test_get_kv_retrieval_prompt_with_query_aware_contextualization():
    data = [("test key1", "test value1"), ("test key2", "test value2"), ("test key3", "test value3")]
    key = "test key2"

    prompt = get_kv_retrieval_prompt(data=data, key=key, query_aware_contextualization=True)
    gold = "\n".join(
        [
            "Extract the value corresponding to the specified key in the JSON object below.",
            "",
            "Key: \"test key2\"",
            "",
            "JSON data:",
            "{\"test key1\": \"test value1\",",
            " \"test key2\": \"test value2\",",
            " \"test key3\": \"test value3\"}",
            "",
            "Key: \"test key2\"",
            "Corresponding value:",
        ]
    )
    assert prompt == gold
