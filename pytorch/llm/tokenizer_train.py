import transformers
from datasets import load_dataset


def train_tokenizer():
    raw_datasets = load_dataset("/home/cx/datas/code_search_net", "python")
    print(raw_datasets)
    print(raw_datasets["train"][123456]["whole_func_string"])

    training_corpus = (
        raw_datasets["train"][i: i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

    def get_training_corpus_v1():
        return (
            raw_datasets["train"][i: i + 1000]["whole_func_string"]
            for i in range(0, len(raw_datasets["train"]), 1000)
        )

    def get_training_corpus_v2():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx: start_idx + 1000]
            yield samples["whole_func_string"]


    old_tokenizer = transformers.AutoTokenizer.from_pretrained('/home/cx/checkpoints/gpt2')
    example = '''def add_numbers(a, b):
        """Add the two numbers `a` and `b`."""
        return a + b'''

    tokens = old_tokenizer.tokenize(example)
    print(len(tokens), tokens)

    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    tokens = tokenizer.tokenize(example)
    print(len(tokens), tokens)

    tokenizer.save_pretrained("/home/cx/output/llm/code-search-net-tokenizer")



def test_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/cx/checkpoints/bert-base")
    example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    encoding = tokenizer(example)
    print(type(encoding))
    print(tokenizer.is_fast)
    print(encoding.tokens())
    print(encoding.word_ids())
    start, end = encoding.word_to_chars(1)
    print(start, end, example[start:end])

    inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
    print(inputs_with_offsets["offset_mapping"])

    from transformers import pipeline
    ckpt_path = "/home/cx/checkpoints/bert-large-cased-finetuned-con1103-english"
    token_classifier = pipeline("token-classification",
                                model=ckpt_path,
                                aggregation_strategy="simple")
    example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    ret = token_classifier(example)
    print(ret)

    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForTokenClassification.from_pretrained(ckpt_path)
    inputs = tokenizer(example, return_tensors="pt")
    outputs = model(**inputs)
    print(inputs["input_ids"].shape)
    print(outputs.logits.shape)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    print(probabilities)
    print(predictions)
    print(model.config.id2label)

    results = []
    tokens = inputs.tokens()
    for idx, pred in enumerate(predictions):
        label = model.config.id2label[pred]
        if label != "O":
            results.append(
                {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
            )
    print(results)
    inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
    print(inputs_with_offsets["offset_mapping"])

    results = []
    inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
    tokens = inputs_with_offsets.tokens()
    offsets = inputs_with_offsets["offset_mapping"]
    for idx, pred in enumerate(predictions):
        label = model.config.id2label[pred]
        if label != "O":
            start, end = offsets[idx]
            results.append(
                {
                    "entity": label,
                    "score": probabilities[idx][pred],
                    "word": tokens[idx],
                    "start": start,
                    "end": end,
                }
            )
    print(results)

    import numpy as np
    results = []
    inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
    tokens = inputs_with_offsets.tokens()
    offsets = inputs_with_offsets["offset_mapping"]
    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = model.config.id2label[pred]
        if label != "O":
            # Remove the B- or I-
            label = label[2:]
            start, _ = offsets[idx]
            # Grab all the tokens labeled with I-label
            all_scores = []
            while (
                    idx < len(predictions)
                    and model.config.id2label[predictions[idx]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx][pred])
                _, end = offsets[idx]
                idx += 1
            # The score is the mean of all the scores of the tokens in that grouped entity
            score = np.mean(all_scores).item()
            word = example[start:end]
            results.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(results)



def qa_test():
    from transformers import pipeline

    ckpt_path = "/home/cx/checkpoints/distilbert-base-cased-distiled-squad"
    question_answerer = pipeline("question-answering", model=ckpt_path)
    context = """
    🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
    between them. It's straightforward to train your models with one before loading them for inference with the other.
    """
    question = "Which deep learning libraries back 🤗 Transformers?"
    ret = question_answerer(question=question, context=context)
    print(ret)

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)

    inputs = tokenizer(question, context, return_tensors="pt")
    print(inputs.tokens())
    outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    print(start_logits.shape, end_logits.shape)

    import torch
    sequence_ids = inputs.sequence_ids()
    print(sequence_ids)
    # Mask everything apart from the tokens of the context
    mask = [i != 1 for i in sequence_ids]
    # Unmask the [CLS] token
    mask[0] = False
    mask = torch.tensor(mask)[None]
    print(mask)
    start_logits[mask] = -10000
    end_logits[mask] = -10000

    start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
    end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
    scores = start_probabilities[:, None] * end_probabilities[None, :]
    scores = torch.triu(scores)
    print(scores)
    max_index = scores.argmax().item()
    start_index = max_index // scores.shape[1]
    end_index = max_index % scores.shape[1]
    print(scores[start_index, end_index])

    inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
    offsets = inputs_with_offsets["offset_mapping"]
    print(offsets)
    start_char, _ = offsets[start_index]
    _, end_char = offsets[end_index]
    answer = context[start_char:end_char]
    result = {
        "answer": answer,
        "start": start_char,
        "end": end_char,
        "score": scores[start_index, end_index],
    }
    print(result)

    sentence = "This sentence is not too long but we are going to split it anyway."
    inputs = tokenizer(
        sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
    )
    for ids in inputs["input_ids"]:
        print(tokenizer.decode(ids))
    print(inputs.keys())
    print(inputs["overflow_to_sample_mapping"])
    print(inputs)

    long_context = """
[CLS] Which deep learning libraries back [UNK] Transformers? [SEP] [UNK] Transformers : State of the Art NLP

[UNK] Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

[UNK] Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internal [SEP]
"""
    inputs = tokenizer(
        question,
        long_context,
        stride=128,
        max_length=384,
        padding="longest",
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    for ids in inputs["input_ids"]:
        print(tokenizer.decode(ids))

    _ = inputs.pop("overflow_to_sample_mapping")
    offsets = inputs.pop("offset_mapping")
    print('offsets: ', len(offsets[0]), len(offsets[1]), offsets)

    inputs = inputs.convert_to_tensors("pt")
    print(inputs["input_ids"].shape)
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    print(start_logits.shape, end_logits.shape)

    sequence_ids = inputs.sequence_ids()
    # Mask everything apart from the tokens of the context
    mask = [i != 1 for i in sequence_ids]
    # Unmask the [CLS] token
    mask[0] = False
    # Mask all the [PAD] tokens
    mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

    start_logits[mask] = -10000
    end_logits[mask] = -10000
    start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
    end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

    candidates = []
    for start_probs, end_probs in zip(start_probabilities, end_probabilities):
        scores = start_probs[:, None] * end_probs[None, :]
        idx = torch.triu(scores).argmax().item()

        start_idx = idx // scores.shape[1]
        end_idx = idx % scores.shape[1]
        score = scores[start_idx, end_idx].item()
        candidates.append((start_idx, end_idx, score))

    print(candidates)
    for candidate, offset in zip(candidates, offsets):
        start_token, end_token, score = candidate
        start_char, _ = offset[start_token]
        _, end_char = offset[end_token]
        answer = long_context[start_char:end_char]
        result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
        print(result)
        

if __name__ == '__main__':
    train_tokenizer()
