import torch
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import evaluate


class NERPipeline:
    def __init__(self, pretrained_model="bert-base-multilingual-cased"):
        self.pretrained_model = pretrained_model
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model)
        self.metric = evaluate.load("seqeval")
        self.label_list = None
        self.label_to_id = None
        self.id_to_label = None
        self.model = None

    def load_model(self, model_dir="./bio_ner_results/best_model"):
        self.model = BertForTokenClassification.from_pretrained(model_dir, use_safetensors=True)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        print(f"Model loaded from: {model_dir}")

        
    def load_data(self, file_path):
        sentences, labels = [], []
        current_sentence, current_labels = [], []

        with open(file_path, "r") as file:
            for line in file:
                if line.strip() == "":
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence, current_labels = [], []
                else:
                    token, label = line.strip().split("\t")
                    current_sentence.append(token)
                    current_labels.append(label)

        return sentences, labels

    def prepare_data(self, train_path, val_path, test_path):
        train_sentences, train_labels = self.load_data(train_path)
        val_sentences, val_labels = self.load_data(val_path)
        test_sentences, test_labels = self.load_data(test_path)

        self.label_list = sorted(set(label for labels in train_labels + val_labels + test_labels for label in labels))
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        def convert_to_hf_format(sentences, labels):
            return Dataset.from_dict({
                "tokens": sentences,
                "ner_tags": [[self.label_to_id[label] for label in label_seq] for label_seq in labels]
            })

        train_dataset = convert_to_hf_format(train_sentences, train_labels)
        val_dataset = convert_to_hf_format(val_sentences, val_labels)
        test_dataset = convert_to_hf_format(test_sentences, test_labels)

        train_dataset = train_dataset.map(self.tokenize_and_align_labels, batched=True, remove_columns=["tokens", "ner_tags"])
        val_dataset = val_dataset.map(self.tokenize_and_align_labels, batched=True, remove_columns=["tokens", "ner_tags"])
        test_dataset = test_dataset.map(self.tokenize_and_align_labels, batched=True, remove_columns=["tokens", "ner_tags"])

        return train_dataset, val_dataset, test_dataset

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            padding=True,
            truncation=True,
            max_length=128,  
            is_split_into_words=True
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def initialize_model(self, num_labels):
        self.model = BertForTokenClassification.from_pretrained(self.pretrained_model, num_labels=num_labels)

    def compute_metrics(self, pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = [
            [self.label_list[l] for l in label if l != -100]
            for label in labels
        ]
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, train_dataset, val_dataset, output_dir="./bio_ner_results"):
        num_epochs = int(input("Enter the number of epochs for training: "))

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            report_to="none",
            logging_dir="./logs",
            logging_steps=10000,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        # Save model and tokenizer
        model_save_path = f"{output_dir}/best_model"
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)  # Save tokenizer here
        print(f"Model and tokenizer saved at: {model_save_path}")


    def predict(self, sentence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        predicted_labels = [self.id_to_label[pred] for pred in predictions[0].cpu().numpy()]
        return list(zip(tokens, predicted_labels))

    def test(self, test_dataset):
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            args=TrainingArguments(
                output_dir="./bio_ner_results",
                report_to="none",
                eval_strategy="no",
            ),
        )

        results = trainer.predict(test_dataset)
        logits = results.predictions
        labels = results.label_ids

        predictions = np.argmax(logits, axis=2)
        true_labels = [[self.label_list[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metrics = self.metric.compute(predictions=true_predictions, references=true_labels)
        return metrics
