import os
import torch
import transformers
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import evaluate
from evaluate.module import EvaluationModule
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np

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


def trainer_test():
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    raw_datasets = load_dataset(data_path, "mrpc")
    def tokenize_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True)

    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(bert_cased_path, num_labels=2)
    def compute_metrics(eval_preds):
        # metric = evaluate.evaluator("text-classification")
        metric = evaluate.load("../../evaluate-main/metrics/accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments('test-trainer', evaluation_strategy="epoch")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    predicts = trainer.predict(tokenized_dataset["validation"])
    print(predicts.predictions.shape, predicts.label_ids.shape)
    print(type(predicts.label_ids))


if __name__ == '__main__':
    # tokenizer_test()
    mrpc_test()
