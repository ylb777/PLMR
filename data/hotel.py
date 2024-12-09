import os
import csv
import random
import torch
from torch.utils.data import Dataset


class HotelDataset(Dataset):
    def __init__(self, tokenizer, data_dir, aspect, mode, max_length, balance):
        super(HotelDataset, self).__init__()
        self.aspect_mapping = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        # Set file path for the dataset
        self.file_path = os.path.join(data_dir, f'hotel_{self.aspect_mapping[aspect]}.{mode}')

        # Create examples and tokenize
        examples_text = self._create_examples(self._read_csv(self.file_path), mode, balance)
        self._tokenize_examples(tokenizer, examples_text, max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "label": self.labels[index]
        }

    def _tokenize_examples(self, tokenizer, examples_text, max_length):
        tokenized_data = tokenizer(examples_text, padding="max_length", max_length=max_length, truncation=True)
        self.input_ids = torch.tensor(tokenized_data['input_ids'], dtype=torch.long)
        self.attention_masks = torch.tensor(tokenized_data['attention_mask'], dtype=torch.long)

    def _read_csv(self, file_path, quotechar=None):
        """Reads a tab-separated value file."""
        with open(file_path, "rt", encoding='utf-8') as file:
            reader = csv.reader(file, delimiter="\t", quotechar=quotechar)
            return [line for line in reader]

    def _create_examples(self, lines, mode, balance=False):
        examples = []
        texts = []

        # Skip the header line and process the remaining lines
        for i, line in enumerate(lines):
            if i == 0:
                continue
            label = int(line[1])
            text = line[2].strip()
            examples.append({'text': text, 'label': label})

        positive_examples = [example for example in examples if example['label'] == 1]
        negative_examples = [example for example in examples if example['label'] == 0]

        # Balance the dataset if specified
        if balance:
            min_count = min(len(positive_examples), len(negative_examples))
            positive_examples = random.sample(positive_examples, min_count)
            negative_examples = random.sample(negative_examples, min_count)
            examples = positive_examples + negative_examples

        for example in examples:
            self.labels.append(example['label'])
            texts.append(example['text'])

        return texts


class HotelAnnotationData(Dataset):
    def __init__(self, tokenizer, data_dir, aspect, max_length):
        super(HotelAnnotationData, self).__init__()
        self.aspect_mapping = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.rationale_masks = []
        self.tokenizer = tokenizer

        aspect_name = self.aspect_mapping[aspect]
        annotation_path = os.path.join(data_dir, f'hotel_{aspect_name}.train')
        lines = self._read_tsv(annotation_path)
        self._create_examples(lines, max_length)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "label": self.labels[index],
            "rationale_mask": self.rationale_masks[index]
        }

    def __len__(self):
        return len(self.labels)

    def _read_tsv(self, annotation_path, quotechar=None):
        """Reads a tab-separated value file."""
        with open(annotation_path, "rt", encoding='utf-8') as file:
            reader = csv.reader(file, delimiter="\t", quotechar=quotechar)
            lines = [line for line in reader]
        return lines

    def _create_examples(self, lines, max_length):
        """
        Create examples from the input data.
        """
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        rationale_masks_list = []

        for i, line in enumerate(lines):
            if i == 0:
                continue  # Skip header

            text_tokens = line[2].split(" ")
            label = int(line[1])
            rationale = [int(x) for x in line[3].split(" ")]

            # Tokenize text without adding special tokens
            tokenized_words = self.tokenizer(text_tokens, add_special_tokens=False)['input_ids']
            token_ids = [token for sublist in tokenized_words for token in sublist]
            attention_mask = [1] * len(token_ids)

            # Create token-level rationale mask
            token_rationale_mask = []
            for idx, word_mask in enumerate(rationale):
                token_rationale_mask += [word_mask] * len(tokenized_words[idx])

            # Padding
            if len(token_ids) <= (max_length - 2):
                padded_input_ids = [101] + token_ids + [102] + [0] * (max_length - 2 - len(token_ids))
                padded_rationale_mask = [0] + token_rationale_mask + [0] + [0] * (max_length - 2 - len(token_rationale_mask))
                padded_attention_mask = [1] + attention_mask + [1] + [0] * (max_length - 2 - len(attention_mask))
            else:
                padded_input_ids = [101] + token_ids[:max_length - 2] + [102]
                padded_rationale_mask = [0] + token_rationale_mask[:max_length - 2] + [0]
                padded_attention_mask = [1] + attention_mask[:max_length - 2] + [1]

            input_ids_list.append(padded_input_ids)
            labels_list.append(label)
            attention_masks_list.append(padded_attention_mask)
            rationale_masks_list.append(padded_rationale_mask)

        self.input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks_list, dtype=torch.long)
        self.labels = torch.tensor(labels_list, dtype=torch.long)
        self.rationale_masks = torch.tensor(rationale_masks_list, dtype=torch.long)
