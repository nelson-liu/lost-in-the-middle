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

The `ordered_kv_records` is a list of [key, value] pairs. The `key` specifies a particular key to retrieve from `ordered_kv_records`, and the `value` lists its expected associated value.

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
