import torch
import transformers
from datasets import load_dataset
from transformers import DataCollatorWithPadding

bert_cased_path = "/home/cx/checkpoints/bert-base"
data_path = "/home/cx/datas"


def tokenizer_test():
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    inputs = tokenizer("This is the first sentence.", "This is the second one.")
    print(inputs)
    print(tokenizer.encode("This is the first sentence.", "This is the second one."))
    print(tokenizer.encode_plus("This is the first sentence.", "This is the second one."))
    print(tokenizer.decode(inputs["input_ids"]))
    sentences = ["God bless you!", "You can do it."]
    print(tokenizer(sentences))
    ids = tokenizer.encode(sentences)
    print(ids)
    print(tokenizer.decode(ids))


def mrpc_test():
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    raw_datasets = load_dataset(data_path, "mrpc")
    raw_train_dataset = raw_datasets['train']
    print(raw_train_dataset[0])
    print(raw_train_dataset.features)
    sentence1 = raw_datasets['train']['text1']
    print(len(sentence1))
    # print(sentence1)
    sentence2 = raw_datasets['train']['text2']
    print(len(sentence2))
    # print(sentence2)
    # tokenizer_dataset = tokenizer(raw_train_dataset["text1"], raw_train_dataset["text2"], padding=True, truncation=True)
    # print(tokenizer_dataset.keys())

    def tokenize_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True)

    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)
    print(tokenized_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    samples = tokenized_dataset["train"][:10]
    # remove the string column or the dynamic padding would fail
    samples = {k:v for k, v in samples.items() if k not in ["idx", "text1", "text2", "label_text"]}
    print(samples)
    print([len(x) for x in samples['input_ids']])
    batch = data_collator(samples)
    print({k:v.shape for k, v in batch.items()})

    samples_validation = tokenized_dataset["validation"][:6]
    samples_validation = {k: v for k, v in samples_validation.items() if k not in ["idx", "text1", "text2", "label_text"]}
    print([len(x) for x in samples_validation['input_ids']])
    batch = data_collator(samples_validation)
    print({k: v.shape for k, v in batch.items()})


if __name__ == '__main__':
    # tokenizer_test()
    mrpc_test()
