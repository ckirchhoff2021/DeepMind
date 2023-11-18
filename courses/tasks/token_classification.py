import transformers
from datasets import load_dataset

from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm.auto import tqdm
import numpy as np
from accelerate import Accelerator

dataset_path = "/home/cx/datas/conll2003"
model_path = "/home/cx/checkpoints/bert-base"
output_path = "/home/cx/output/ner"

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

import numpy as np


def main():
    raw_datasets = load_dataset(dataset_path)
    print(raw_datasets)
    print(raw_datasets["train"][0])
    print(raw_datasets["train"][0]["tokens"])
    print(raw_datasets["train"][0]["ner_tags"])
    ner_feature = raw_datasets["train"].features["ner_tags"]
    print(ner_feature)
    label_names = ner_feature.feature.names
    print("label_names: ", label_names)

    words = raw_datasets["train"][0]["tokens"]
    labels = raw_datasets["train"][0]["ner_tags"]
    line1 = ""
    line2 = ""
    for word, label in zip(words, labels):
        full_label = label_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)

    print(line1)
    print(line2)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
    print(inputs.tokens())
    print(inputs.word_ids())

    labels = raw_datasets["train"][0]["ner_tags"]
    word_ids = inputs.word_ids()
    print(labels)
    print(align_labels_with_tokens(labels, word_ids))

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    # define metric
    import evaluate
    metric = evaluate.load("../../../evaluate-main/metrics/seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    # collate
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
    print(batch["labels"])

    # no collate
    for i in range(2):
        print(tokenized_datasets["train"][i]["labels"])
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    # models
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
    )
    print(model.config.num_labels)

    '''
    args = TrainingArguments(
        "bert-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    '''
    custom_training(tokenized_datasets, label_names, id2label, label2id, tokenizer, data_collator, metric)


def test_metric():
    raw_datasets = load_dataset(dataset_path)
    import evaluate
    metric = evaluate.load("../../../evaluate-main/metrics/seqeval")
    labels = raw_datasets["train"][0]["ner_tags"]
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    labels = [label_names[i] for i in labels]
    predictions = labels.copy()
    predictions[2] = "O"
    ret = metric.compute(predictions=[predictions], references=[labels])
    print(ret)


def postprocess(predictions, labels, label_names):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def custom_training(tokenized_datasets, label_names, id2label, label2id, tokenizer, data_collator, metric):

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
    )
    optimizer = AdamW(model.parameters(), lr=2e-5)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered, label_names)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )
        # Save and upload
        output_dir = output_path
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)


def test_model():
    from transformers import pipeline
    model_checkpoint = '/home/cx/A100/DeepMind/pytorch/nlp/tasks/bert-finetuned-ner/checkpoint-440'
    # model_checkpoint = output_path
    token_classifier = pipeline(
        "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    )
    ret = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(ret)


if __name__ == '__main__':
    main()
    # test_metric()
    # test_model()
