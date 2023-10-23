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


if __name__ == '__main__':
    train_tokenizer()
