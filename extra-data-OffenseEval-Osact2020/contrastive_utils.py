import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn 
from utils import getData
from datetime import datetime
from tqdm.notebook import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datasets import load_metric
from datasets import load_dataset 
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, evaluation
device = torch.device("cuda")

def run_batch_loss_expirement(loss, batch_size, epochs, expirement_name):
    train, test = getData(sub_task="A", return_type="pandas", pre_proccessed=True)
    train_set = make_labeled_dataset(train)
    test_set = make_labeled_dataset(test)
    test_triplets = triplets_from_labeled_dataset(test_set)

    train_data_sampler = SentenceLabelDataset(train_set)
    train_dataloader = DataLoader(train_data_sampler, batch_size=batch_size, drop_last=True) 

    test_evaluator = TripletEvaluator.from_input_examples(test_triplets, name='dev')

    model = SentenceTransformer("UBC-NLP/MARBERTv2")
    train_loss = loss(model=model)

    warmup_steps = int(len(train_dataloader) * epochs  * 0.1)  # 10% of train data
   
    model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=test_evaluator,
                epochs=epochs,
                evaluation_steps=1000,
                warmup_steps=warmup_steps,
                output_path=f"./Checkpoints/SentenceBert/{expirement_name}",
                save_best_model = True
            )
    return train, test, model 

def plot_representation(model,data, batch_size=32):
    pos_samples = model.encode(sentences = list(data[data.labels == 1].text.values), 
                        show_progress_bar = True ,
                        batch_size = batch_size,
                        device=device)
    neg_samples = model.encode(sentences = list(data[data.labels == 0].text.values), 
                        show_progress_bar = True ,
                        batch_size = batch_size,
                        device=device)
    X = np.concatenate((pos_samples, neg_samples))
    y = np.concatenate((data[data.labels == 1].labels.values, data[data.labels == 0].labels.values))
    clf = PCA(n_components=2)
    pca_trans = clf.fit_transform(X)
    print(clf.explained_variance_ratio_.cumsum())
    sns.scatterplot(x=pca_trans[:,0],y=pca_trans[:,1], hue=y, alpha=0.1)


def make_contranstive_data(data, size: str):
    """
    size = "10K"
    """
    from itertools import combinations, product 
    from random import sample
    size = int(size.split("K")[0]) * 1000 #total size of dataset 
    label_0_size = size // 2  #total size of negative examples (half)
    label_1_size = size // 2 #total size of positive examples (half)
    label_1_from_each = label_1_size // 2  #total size of positive examples within each class 

    pos_indexes = data[data.labels == 1].index.values #class 1 indexes
    neg_indexes = data[data.labels == 0].index.values #class 0 indexes 

    label_0_pool = list(product(pos_indexes, neg_indexes)) #cartesian product between class_1_indexes and class_0_indexes to produce all candidate negative examples 
    label_pos_1_pool = list(combinations(pos_indexes,2))  #combination of 2 from class_1_indexes to produce all candidate positive examples from class 1 
    label_neg_1_pool = list(combinations(neg_indexes,2)) #combination of 2 from class_0_indexes to produce all candidate positive examples from class 0 

    assert size <= len(label_0_pool) + len(label_pos_1_pool) + len(label_neg_1_pool), ""
    dataset = []

    label_0_indexes = sample(label_0_pool, k = label_0_size ) #sampling negative examples 
    label_1_indexes = sample(label_pos_1_pool, k = label_1_from_each ) + sample(label_neg_1_pool, k = label_1_from_each ) #positive examples equally from both classes 
    dataset = []
    zero_label = 0
    one_label = 0
    for ind_1, ind_2 in tqdm(label_0_indexes) : 
        dataset.append(InputExample(texts=[data.iloc[ind_1].text, data.iloc[ind_2].text], label=int(0)))
        zero_label+=1
    for ind_1, ind_2 in tqdm(label_1_indexes) : 
        dataset.append(InputExample(texts=[data.iloc[ind_1].text, data.iloc[ind_2].text], label=int(1)))
        one_label+=1


    print(f"Total Data Length {len(dataset)}")
    print(f"Ratio Between Negative and Positive Samples = {zero_label/one_label}")

    return dataset

def make_labeled_dataset(data):
    dataset = []
    guid = 1 
    for text, label in tqdm(zip(data.text.values, data.labels.values), total=len(data.text.values)):
        dataset.append(InputExample(guid= guid ,texts=[text], label=int(label)))
        guid+= 1 
    return dataset

def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets

def run_linearclassifier(model_checkpoint, output_dir, batch_size=64, epochs=100, es_patience = 10, my_classifier_dropout=0.3):  
    f1 = load_metric("f1")
    recall = load_metric("recall")
    precision =  load_metric("precision")
    def preprocess_function(examples, tok):
        return tok(examples["text"], truncation=True, max_length=512)
    def compute_metrics(p):    
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        metric = f1.compute(predictions=predictions, references=labels, average="macro")
        metric.update(recall.compute(predictions=predictions, references=labels, average="macro"))
        metric.update(precision.compute(predictions=predictions, references=labels, average="macro"))
        return metric
    data = getData(sub_task = f"A", return_type = "dataset", pre_proccessed = True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_data = data.map(preprocess_function,fn_kwargs = {'tok':tokenizer}, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, classifier_dropout = my_classifier_dropout)
    model.classifier = nn.Sequential(
                        nn.Linear(768, 512),
                        nn.ReLU(),
                        nn.Dropout(my_classifier_dropout),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 2))
    for name, param in model.named_parameters():
        if 'classifier' not in name : 
            param.requires_grad = False
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-03,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        metric_for_best_model="f1",
        num_train_epochs=100,
        load_best_model_at_end=True,
        group_by_length = True, 
        report_to = 'none'
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=es_patience)]
    )
    return trainer