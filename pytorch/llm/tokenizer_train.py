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


if __name__ == '__main__':
    train_tokenizer()
