
import torch
import numpy as np
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
f1 = load_metric("f1")
recall = load_metric("recall")
precision =  load_metric("precision")

def getData(sub_task, return_type, pre_proccessed=False):
    """
    sub_task : string in 'A', 'B', 'C'
    type : 'pandas' or 'dataset'
    if pandas : 
        returns train, test 
    if dataset: 
        returns dataset[['train', 'test']]
    """
    assert sub_task in ["A", "B", "C"] 
    assert return_type in ["pandas", "dataset"]
    if return_type == 'pandas' :
        if pre_proccessed:
            return pd.read_csv(f"Data/train{sub_task}_prepro.csv", header=0), pd.read_csv(f"Data/test{sub_task}_prepro.csv", header=0)
        else : 
            return pd.read_csv(f"Data/train{sub_task}.csv", header=0), pd.read_csv(f"Data/test{sub_task}.csv", header=0)
    else :
        if pre_proccessed:
            print("Pre-Processed")
            return load_dataset("csv", data_files={'train': f"Data/train{sub_task}_prepro.csv", 'test': f"Data/test{sub_task}_prepro.csv" } )
        else : 
            print("Not Pre-Processed")
            return load_dataset("csv", data_files={'train': f"Data/train{sub_task}.csv", 'test': f"Data/test{sub_task}.csv" } )



def preprocess_function(examples, tok):
    return tok(examples["text"], truncation=True, max_length=512)


def compute_metrics(p):    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    metric = f1.compute(predictions=predictions, references=labels, average="macro")
    metric.update(recall.compute(predictions=predictions, references=labels, average="macro"))
    metric.update(precision.compute(predictions=predictions, references=labels, average="macro"))
    return metric

def run_baseline(model_name, model_link, pre_proccessed, patience, seed, task, epochs = 20 ,dropout_ratio=0.1):
    data = getData(sub_task = f"{task}", return_type = "dataset", pre_proccessed = pre_proccessed)
    tokenizer = AutoTokenizer.from_pretrained(model_link)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_data = data.map(preprocess_function,fn_kwargs = {'tok':tokenizer}, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_link, num_labels=2, classifier_dropout=dropout_ratio)
    torch.manual_seed(seed)
    if pre_proccessed :
        output_dir = f"./Checkpoints/Pre_processed/{model_name}_d_{dropout_ratio}"
    else : 
        output_dir = f"./Checkpoints/{model_name}_d_{dropout_ratio}"
    training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy = "epoch",
    metric_for_best_model="f1",
    num_train_epochs=epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    group_by_length = True, 
    seed=seed
)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    return trainer
