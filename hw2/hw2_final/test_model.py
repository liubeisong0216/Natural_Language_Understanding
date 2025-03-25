"""
Code for Problem 1 of HW 2.
"""
import pickle
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset, compute_metrics

def init_tester(directory: str) -> Trainer:
    """
    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set.
    """
    model = BertForSequenceClassification.from_pretrained(directory)

    training_args = TrainingArguments(do_train=False, evaluation_strategy="no")

    tester = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    return tester

# def init_tester(directory: str) -> Trainer:
#     """
#     Prolem 2b: Implement this function.

#     Creates a Trainer object that will be used to test a fine-tuned
#     model on the IMDb test set. The Trainer should fulfill the criteria
#     listed in the problem set.

#     :param directory: The directory where the model being tested is
#         saved
#     :return: A Trainer used for testing
#     """
#     model = BertForSequenceClassification.from_pretrained(directory)
#     training_args = TrainingArguments(do_train=False, evaluation_strategy="no")


#     tester = Trainer(
#         model=model,
#         args=training_args,
#         compute_metrics=compute_metrics,
#     )

#     return tester



if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("checkpoints_without_bitfit/run-16/checkpoint-1252")
    # tester = init_tester("checkpoints_with_bitfit/run-1/checkpoint-5000")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results_without_bitfit.p", "wb") as f:
        pickle.dump(results, f)