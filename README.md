# Named_Entity_Recognition_BERT_Multilingual_Library_LUX

[![Downloads](https://static.pepy.tech/badge/named-entity-recognition-bert-multilingual-library-lux)](https://pepy.tech/project/named-entity-recognition-bert-multilingual-library-lux)


## Overview

**Named_Entity_Recognition_BERT_Multilingual_Library_LUX** is a powerful and flexible library for Named Entity Recognition (NER) tasks using BERT models. This library supports multilingual NER and is suitable for key information extraction tasks across various domains such as biomedical, environmental, and technological.

The library simplifies NER tasks by providing an easy-to-use pipeline for loading data, training models, and making predictions. It is designed for developers, researchers, and data scientists looking for a robust NER solution.
You can find an example of employing the library at [this Kaggle notebook](https://www.kaggle.com/code/mehrdadal/named-entity-recognition-bert-multilingual-library).


## Features

- **Multilingual Support**: Leverage the power of BERT for NER tasks in multiple languages.
- **Flexible Input Format**: Works with CoNLL format data.
- **Easy Integration**: Provides a simple `NERPipeline` class for seamless integration into your projects.
- **Comprehensive Metrics**: Evaluate your models with precision, recall, F1-score, and accuracy.
- **Pretrained Models**: Supports any BERT-based pretrained models.

## Installation

Install the library using `pip`:
```bash
pip install named-entity-recognition-bert-multilingual-library-lux
```

## Usage

### Example Usage

Here is a complete example of how to use the library for training and predicting:

```bash
# Clone the dataset
!git clone https://github.com/spyysalo/bc2gm-corpus.git
```

```python
from Named_Entity_Recognition_BERT_Multilingual_Library_LUX import NERPipeline

# Initialize pipeline
pipeline = NERPipeline(pretrained_model="bert-base-multilingual-cased")

# Prepare data
train_dataset, val_dataset, test_dataset = pipeline.prepare_data(
    "./bc2gm-corpus/conll/train.tsv", 
    "./bc2gm-corpus/conll/devel.tsv", 
    "./bc2gm-corpus/conll/test.tsv"
)


# Initialize model
pipeline.initialize_model(num_labels=len(pipeline.label_list))

# Train the model
pipeline.train(train_dataset, val_dataset,output_dir="/kaggle/temp/bio_ner_results/")

# Load the model after training
pipeline.load_model(model_dir="/kaggle/temp/bio_ner_results/best_model/")

# Test the model
test_metrics = pipeline.test(test_dataset)
print(test_metrics)

# Predict on a new sentence
predictions = pipeline.predict("BRCA1 is a gene associated with breast cancer.")
print("\nPredictions on New Sentence:")
for token, label in predictions:
    print(f"{token} | {label}")

```

## CoNLL Data Format

The input data should be in CoNLL format:

```mathematica

Token1    Label1
Token2    Label2
Token3    Label3
Token4    Label4
```

### Example:

```mathematica
BRCA1       B-GENE
is          O
a           O
gene        O
associated  O
with        O
breast      B-DISEASE
cancer      I-DISEASE
.           O
```

## Key Components

### 1. NERPipeline

The main class providing methods for:

- **Data Preparation**: Converts CoNLL format to a dataset suitable for BERT.
- **Model Initialization**: Loads a pretrained BERT model for NER.
- **Training**: Fine-tunes the model on your data.
- **Prediction**: Predicts labels for new sentences.
- **Evaluation**: Computes evaluation metrics (precision, recall, F1, etc.).

### 2. Helper Functions

The library also includes utility functions for advanced users:

- **`load_data`**: Load and parse CoNLL format data.
- **`convert_to_hf_format`**: Convert data to Hugging Face dataset format.
- **`compute_metrics`**: Evaluate predictions using `seqeval`.

## Evaluation Metrics

The library uses the `seqeval` library to compute the following metrics:

- Precision
- Recall
- F1-Score
- Accuracy

## Dependencies

- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `numpy`
- `seqeval`

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests.

