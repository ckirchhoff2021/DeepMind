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


def full_train():
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    raw_datasets = load_dataset(data_path, "mrpc")
    def tokenize_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True)

    tokenized_dataset = raw_datasets.map(tokenize_function, remove_columns=['text1', "text2", "idx", "label_text"], batched=True)
    # tokenized_dataset.remove_columns(['text1', "text2", "idx","label_text"])  *** useless ***
    tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    print(tokenized_dataset['train'].column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_dataset['validation'], shuffle=True, batch_size=8, collate_fn=data_collator)
    for batch in train_dataloader:
        print({k: v.shape for k,v in batch.items()})
        break
    model = AutoModelForSequenceClassification.from_pretrained(bert_cased_path, num_labels=2)
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    accelerator = Accelerator()
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model,
                                                                              optimizer)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))
    metric = evaluate.load("../../evaluate-main/metrics/accuracy")
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    metrics = metric.compute()
    print(metrics)


if __name__ == '__main__':
    # tokenizer_test()
    mrpc_test()
