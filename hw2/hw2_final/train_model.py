"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict
import torch

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype="auto")
    if use_bitfit:
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    return model



def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    
    """
    training_args = TrainingArguments(
        output_dir="checkpoints_with_bitfit",            
        num_train_epochs=4,                  
        per_device_train_batch_size=8,       
        per_device_eval_batch_size=8,        
        evaluation_strategy="epoch",         
        save_strategy="epoch",               
        load_best_model_at_end=True,       
        metric_for_best_model="eval_accuracy",    
        greater_is_better=True,              
        seed=3463                            
    )


    trainer = Trainer(
        model_init=lambda: init_model(None, model_name, use_bitfit=use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    return trainer



def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Returns keyword arguments to pass to Trainer.hyperparameter_search.
    We define a full grid over:
    
    - lr: [3e-4, 1e-4, 5e-5, 3e-5]
    - batch_size: [8, 16, 32, 64, 128]
    
    This ensures every combination is tested (4 * 5 = 20 trials).
    We also set the search direction to "maximize" for accuracy.
    """

    
    search_space = {
        'learning_rate': [3e-4, 1e-4, 5e-5, 3e-5],
        'per_device_train_batch_size': [8, 16, 32, 64, 128]
    }

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", search_space['learning_rate']),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", search_space['per_device_train_batch_size'])
        }

    
    return {
        "sampler": optuna.samplers.GridSampler(search_space),
        "direction": "maximize",
        "backend": "optuna",
        "n_trials": 20,
        "hp_space": optuna_hp_space 
    }




if __name__ == "__main__":  # Use this script to train your model
    print(torch.cuda.is_available())
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best, f)
