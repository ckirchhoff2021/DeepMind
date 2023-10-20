import os
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding
import evaluate
from evaluate.module import EvaluationModule
from transformers import Trainer, AutoModelForSequenceClassification, AdamW, get_scheduler
from transformers import TrainingArguments
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import html


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
    model.save_pretrained("/home/cx/output/llm")
    tokenizer.save_pretrained("/home/cx/output/llm")

  
def test_dataset():
    # squad_it_dataset = load_dataset("json", data_files="/home/cx/datas/SQuAD_it-train.json.gz", field="data")
    # print(squad_it_dataset)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path, use_fast=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    def tokenize_function(example):
        return tokenizer(example["review"])

    data_files = {"train": "/home/cx/datas/drugsComTrain_raw.tsv", "test": "/home/cx/datas/drugsComTest_raw.tsv"}
    drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
    # print(drug_sample[:3])
    print(drug_dataset.keys())

    for split in drug_dataset.keys():
        assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
    drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", new_column_name="patient_id")

    def lowercase_condition(example):
        return {"condition": example["condition"].lower()}

    drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
    drug_dataset = drug_dataset.map(lowercase_condition)
    drug_dataset = drug_dataset.map(lambda x: {"review_length": len(x["review"].split())})
    print(drug_dataset["train"].sort("review_length")[:3])
    drug_dataset =drug_dataset.filter(lambda x: x["review_length"] > 30)
    print(drug_dataset.num_rows)

    text = "I&#039;m a transformer called BERT"
    html.unescape(text)
    print(text)
    drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
    new_drug_dataset = drug_dataset.map(lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batch_size=True)

    '''
    import time
    start = time.time()
    # drug_dataset = drug_dataset.map(tokenize_function, batched=True, num_proc=8)
    drug_dataset = drug_dataset.map(tokenize_function, batched=True)
    end = time.time()
    print('cost: ', end- start)
    '''

    def tokenize_and_split(example):
        result = tokenizer(example["review"], truncation=True, max_length=128, return_overflowing_tokens=True)
        sample_map = result.pop("overflow_to_sample_mapping")
        for key, values in example.items():
            result[key] = [values[i] for i in sample_map]
        return result

    collate_fn = lambda x: tokenizer(x["review"], truncation=True, max_length=128, return_overflowing_tokens=True)
    result = collate_fn(drug_dataset['train'][0])
    print(drug_dataset['train'][0]['review'])
    print([len(inp) for inp in result["input_ids"]])
    # tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True,
    #                                      remove_columns=drug_dataset['train'].column_names)
    tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)

    print(tokenized_dataset)
    print(len(drug_dataset['train']))
    print(len(tokenized_dataset['train']))

    drug_dataset.set_format("pandas")
    print(drug_dataset["train"][:3])

    train_df = drug_dataset["train"][:]
    frequencies = (
        train_df["condition"]
            .value_counts()
            .to_frame()
            .reset_index()
            .rename(columns={"index": "condition", "condition": "frequency"})
    )
    print(frequencies.head())

    drug_dataset_clean = drug_dataset["train"].train_test_split(traain_size=0.8, seed=42)
    drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
    drug_dataset_clean["test"] = drug_dataset["test"]
    print(drug_dataset_clean)

    # save data
    '''
    drug_dataset_clean.save_to_disk("drug-reviews")
    from datasets import load_from_disk
    drug_dataset_reloaded = load_from_disk("drug-reviews")

    for split, dataset in drug_dataset_clean.items():
        dataset.to_json(f"drug-reviews-{split}.jsonl")
    '''


def test_bigdataset():
    from datasets import load_dataset
    # This takes a few minutes to run, so go grab a tea or coffee while you wait :)
    data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
    pubmed_dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)
    print(pubmed_dataset)


def test_pandas():
    import pandas as pd
    df = pd.DataFrame({
        '姓名': ["小强", "小李", "小王", "张飞"],
        "年龄": [24, 46, 22, 42],
        "籍贯": ["北京", "上海", "广州", "四川"],
        "得分": [[11,12],[21,1,2],[1,1,2,21], [2,3,2]]
    })
    print(df)
    df =  df.explode("得分", ignore_index=True)
    print(df)


def test_faiss():
    issues_dataset = load_dataset("/home/cx/datas/issues", split="train")
    print(issues_dataset)
    issues_dataset = issues_dataset.filter(lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0))
    print(issues_dataset)
    columns = issues_dataset.column_names
    print(columns)
    columns_to_keep = ["title", "body", "html_url", "comments"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    issues_dataset = issues_dataset.remove_columns(columns_to_remove)
    print(issues_dataset[:3])
    issues_dataset.set_format("pandas")
    df = issues_dataset[:]
    print('before explode ...')
    print(df.iloc[0, 3])
    comments_df = df.explode("comments", ignore_index=True)
    print('after explode ...')
    print(comments_df.iloc[0, 3])

    comments_dataset = Dataset.from_pandas(comments_df)
    comments_dataset = comments_dataset.map(
        lambda x: {"comment_length": len(x["comments"].split())}
    )
    comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

    def concatenate_text(examples):
        return {
            "text": examples["title"]
                    + " \n "
                    + examples["body"]
                    + " \n "
                    + examples["comments"]
        }

    comments_dataset = comments_dataset.map(concatenate_text)
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    model = transformers.AutoModel.from_pretrained(bert_cased_path)
    print('**********************')
    print(model)
    print('**********************')
    model.cuda()
    device = torch.device('cuda')

    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(text_list):
        encoded_input = tokenizer(
            text_list, padding=True, max_length=512, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return cls_pooling(model_output)

    text_list = comments_dataset["text"][0]
    print('Text Num: ', len(text_list), type(text_list[0]))
    embedding = get_embeddings(text_list)
    print(embedding.shape)

    embeddings_dataset = comments_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
    )
    embeddings_dataset.add_faiss_index(column="embeddings")
    question = "How can I load a dataset offline?"
    question_embedding = get_embeddings([question]).cpu().detach().numpy()
    print(question_embedding.shape)

    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=5
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()

def bert_embedding():
    model = transformers.AutoModel.from_pretrained(bert_cased_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_cased_path)
    text = ["you x x x x x"] + ["and I"] * 2048
    text = ''.join(text)
    print(len(text), type(text[0]))
    print(text[0])
    ids = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    # print(ids)
    outputs = model(**ids)
    print(outputs)
        

if __name__ == '__main__':
    # tokenizer_test()
    mrpc_test()
