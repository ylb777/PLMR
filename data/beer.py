import gzip
import json
import os
import random

import torch
from torch.utils.data import Dataset

class BeerDataCorrelated(Dataset):
    def __init__(self, tokenizer, data_dir, aspect, data_type, max_length, balance):
        super().__init__()
        self.file_path = os.path.join(data_dir, 'reviews.260k.{}.txt.gz'.format(data_type))
        self.labels = []
        self.inputs = []
        self.attention_masks = []

        # Create examples and tokenize them
        examples_text = self._create_examples( aspect, balance)
        self._tokenize_examples(tokenizer, examples_text, max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.inputs[index],
            "attention_mask": self.attention_masks[index],
            "label": self.labels[index]
        }

    def _tokenize_examples(self, tokenizer, examples_text, max_length):
        tokenized_data = tokenizer(examples_text, padding="max_length", max_length=max_length, truncation=True)
        self.inputs = torch.tensor(tokenized_data['input_ids'], dtype=torch.long)
        self.attention_masks = torch.tensor(tokenized_data['attention_mask'], dtype=torch.long)

    def _create_examples(self, aspect, balance=False):
        """
        Create examples from the input data file based on aspect sentiment scores.
        """
        examples = []
        texts = []

        with gzip.open(self.file_path, "rt") as file:
            lines = file.readlines()
            for line in lines:
                labels, text = line.split('\t')
                labels = [float(value) for value in labels.split()]
                if labels[aspect] <= 0.4:
                    label = 0
                elif labels[aspect] >= 0.6:
                    label = 1
                else:
                    continue
                examples.append({'text': text.strip(), 'label': label})

        print('Beer dataset')
        print(f'samples: {len(examples)}')

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print(f"data: {len(pos_examples)} positive examples, {len(neg_examples)} negative examples.")

        if balance:
            min_examples = min(len(pos_examples), len(neg_examples))
            pos_examples = random.sample(pos_examples, min_examples)
            neg_examples = random.sample(neg_examples, min_examples)
            assert len(pos_examples) == len(neg_examples)
            examples = pos_examples + neg_examples

        for example in examples:
            self.labels.append(example['label'])
            texts.append(example['text'])

        return texts


class BeerAnnotationData(Dataset):
    def __init__(self, tokenizer, annotation_path, aspect, max_length):
        super().__init__()
        self.inputs = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self.tokenizer = tokenizer
        self._create_examples(annotation_path, aspect, max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.inputs[index],
            "attention_mask": self.masks[index],
            "label": self.labels[index],
            "rationale": self.rationales[index]
        }

    def _create_examples(self, annotation_path, aspect, max_length):
        inputs = []
        masks = []
        labels = []
        rationales = []


        with open(annotation_path, "rt", encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)

                # Extract data
                text_tokens = item["x"]
                aspect_label = item["y"][aspect]
                rationale_indices = item[str(aspect)]

                if not rationale_indices:
                    continue  # Skip if no rationale for the given aspect

                # Process the label
                if float(aspect_label) <= 0.4:
                    label = 0
                elif float(aspect_label) >= 0.6:
                    label = 1
                else:
                    continue  # Skip if label is in between thresholds

                # Tokenize text without adding special tokens
                tokenized_words = self.tokenizer(text_tokens, add_special_tokens=False)['input_ids']
                token_ids = [token for sublist in tokenized_words for token in sublist]
                input_mask = [1] * len(token_ids)

                # Create word-level rationale mask
                word_rationale_mask = [0] * len(tokenized_words)
                for start, end in rationale_indices:
                    for idx in range(start, end):
                        word_rationale_mask[idx] = 1

                # Create token-level rationale mask
                token_rationale_mask = [mask for idx, mask in enumerate(word_rationale_mask) for _ in tokenized_words[idx]]

                # Padding
                if len(token_ids) <= (max_length - 2):
                    padded_input_ids = [101] + token_ids + [102] + [0] * (max_length - 2 - len(token_ids))
                    padded_rationale_mask = [0] + token_rationale_mask + [0] + [0] * (max_length - 2 - len(token_rationale_mask))
                    padded_input_mask = [1] + input_mask + [1] + [0] * (max_length - 2 - len(input_mask))
                else:
                    padded_input_ids = [101] + token_ids[:max_length - 2] + [102]
                    padded_rationale_mask = [0] + token_rationale_mask[:max_length - 2] + [0]
                    padded_input_mask = [1] + input_mask[:max_length - 2] + [1]

                inputs.append(padded_input_ids)
                labels.append(label)
                masks.append(padded_input_mask)
                rationales.append(padded_rationale_mask)

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)
        self.rationales = torch.tensor(rationales, dtype=torch.long)


