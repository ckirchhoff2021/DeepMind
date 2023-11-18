import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

import collections
import numpy as np

from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import Trainer
from transformers import get_scheduler

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import math

from accelerate import Accelerator

checkpoint_path = "/home/cx/checkpoints/distilbert-base-uncased"
data_path = "/home/cx/datas/imdb"
output_path = "/home/cx/output/mlm"

wwm_probability = 0.2
def whole_word_masking_data_collator(features, tokenizer):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


def custom_training(downsampled_dataset, eval_dataset, tokenizer, data_collator):
    batch_size = 64
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )
    model = transformers.AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    optimizer = AdamW(model.parameters(), lr=5e-5)

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
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        accelerator.wait_for_everyone()
        output_dir = output_path
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)


def main():
    model = transformers.AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    distilbert_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
    print(f"'>>> BERT number of parameters: 110M'")

    text = "This is a great [MASK]."
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

    imdb_dataset = load_dataset(data_path)
    sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
    for row in sample:
        print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = imdb_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    )
    print(tokenized_datasets)
    print(tokenizer.model_max_length)
    tokenized_samples = tokenized_datasets["train"][:3]

    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Review {idx} length: {len(sample)}'")

    concatenated_examples = {
        k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])
    print(f"'>>> Concatenated reviews length: {total_length}'")

    chunk_size = 128
    chunks = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    for chunk in chunks["input_ids"]:
        print(f"'>>> Chunk length: {len(chunk)}'")

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    print(lm_datasets)
    print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
    print(tokenizer.decode(lm_datasets["train"][1]["labels"]))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    samples = [lm_datasets["train"][i] for i in range(2)]
    for sample in samples:
        _ = sample.pop("word_ids")

    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")

    samples = [lm_datasets["train"][i] for i in range(2)]
    batch = whole_word_masking_data_collator(samples, tokenizer)

    for chunk in batch["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")

    train_size = 10_000
    test_size = int(0.1 * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    '''
    batch_size = 64
    # Show the training loss with every epoch
    logging_steps = len(downsampled_dataset["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir=f"distilbert-base-uncased-finetuned-imdb",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps=logging_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    '''
    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    eval_dataset = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )

    custom_training(downsampled_dataset, eval_dataset, tokenizer, data_collator)


def test_model():
    from transformers import pipeline
    # model_checkpoint = '/home/cx/A100/DeepMind/pytorch/nlp/tasks/bert-finetuned-ner/checkpoint-440'
    model_checkpoint = output_path
    mask_filler = pipeline(
        "fill-mask", model=output_path
    )
    preds = mask_filler("This is a great [MASK].")
    for pred in preds:
        print(f">>> {pred['sequence']}")



if __name__ == '__main__':
    # main()
    test_model()
