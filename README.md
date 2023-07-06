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
