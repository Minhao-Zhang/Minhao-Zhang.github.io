---
title: Fine Tune LLM for Classification
categories:
  - blog
tags:
  - en
  - tutorial
toc: True
---
A step-by-step guide to fine-tuning Large Language Models for text classification tasks

## Introduction

Fine-tuning Large Language Models (LLMs) for specific tasks is one of the most exciting developments in Natural Language Processing (NLP). 
With the rapid advancements in pre-trained LLMs, it is now possible to adapt these models to a wide range of applications, such as text classification, summarization, and sentiment analysis, with minimal effort.

In this guide, we’ll explore how to fine-tune an LLM for text classification using an example task: classifying MBTI personality traits from text data. 
While MBTI is a specific use case, the techniques and workflow discussed here can be generalized to other text classification problems.

## Prerequisites

I am assuming that you have a basic understanding of Python and you have a NVIDIA GPU with CUDA installed. 
If you don’t have a GPU, you might want to consider using a cloud-based service. 
If you have trouble setting up the environment, feel free to check out my [tutorial](../2024-11-15-docker-as-vm). 

## Set up the environment

Using `conda` might be the easiest way to set up the environment.
Create and activate a new conda environment with Python 3.11.9:

```bash
conda create -n llm-tuning python=3.11.9 
conda activate llm-tuning  
```

Feel free to use a different version of Python if you prefer.
However, make sure that the version is compatible with the libraries we are going to use.
Install the required libraries:

```bash
pip install pytorch datasets transformers evaluate accelerate   
```

## Working with Huggingface packages

To save the model, we will need to create an account on the Hugging Face website and get an access key. 
You can find more inforamtion on on [Huggingface](https://huggingface.co/docs/hub/en/security-tokens). 
We can save the access key in a `.env` file in the root directory of your project.

```python
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()
login(token=os.getenv("HF_HUB_TOKEN"))
```

## Load the dataset

Then, you can load the dataset using the `datasets` library.
The following code will either download the dataset from Huggingface or load it from cache. 

```python
from datasets import load_dataset

mbti_data = load_dataset("minhaozhang/mbti", split='train')
```

To ensure we have a good train-validation split, we will be using stratified sampling.
Thus, we need to encode the column first. 
    
```python
mbti_data = mbti_data.class_encode_column("mbti")
mbti_data = mbti_data.class_encode_column("E-I")
mbti_data = mbti_data.class_encode_column("N-S")
mbti_data = mbti_data.class_encode_column("F-T")
mbti_data = mbti_data.class_encode_column("J-P")
```

Since the training split contains about 2.3M rows of data, training on all of them will take too long. 
Thus, we will be using only 1% of the data for training and 20% of that data for validation for this tutorial.

```python
mbti_data = mbti_data.train_test_split(test_size=0.01, stratify_by_column="mbti", seed=0)
mbti_data = mbti_data["test"] # throw away 99% of the data
mbti_data = mbti_data.train_test_split(test_size=0.2, stratify_by_column="mbti", seed=1)
```

## Define the model 

The official Hugging Face guide uses BERT which is quite old at this point. 
Here, I will use the `Phi-3-mini-4k-instruct` model, which is released in 2024. 
You can try other models if you want to like `meta-llama/Llama-3.2-3B-Instruct`. 

```python
MODEL = "microsoft/Phi-3-mini-4k-instruct" 
TRAINED_MODEL = "Phi-3-mini-4k-instruct-mbti"
```

Then, we can define the model and the tokenizer.

```python
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess_function(data):
    return tokenizer(data["body"], truncation=True)

tokenized_mbti_data = mbti_data.map(preprocess_function, batched=True)
del mbti_data # save RAM
```

To simplified the problem, I will only classify the J-P aspect of MBTI as it is somewhat imbalance but not skewed too much. In this dataset, the J-P aspect has a 60-40 split.Hence, I will remove all other columns.

```python
tokenized_mbti_data = tokenized_mbti_data.remove_columns(['author', 'mbti', 'F-T', "E-I", 'N-S'])
tokenized_mbti_data = tokenized_mbti_data.rename_column('J-P', "label")
```

To simplify batching data and improve efficiency by padding tokens on the fly, we will use a `DataCollator`. 

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

To evaluate our model, the `evaluate` package provided a lot of out-of-box metrics we can use. 

```python
import evaluate

# use 3 different metrics to evaluate the model
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
balanced_accuracy = evaluate.load("hyperml/balanced_accuracy")
matthews_correlation = evaluate.load("matthews_correlation")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)
    balanced_accuracy_result = balanced_accuracy.compute(predictions=predictions, references=labels)
    matthews_correlation_result = matthews_correlation.compute(predictions=predictions, references=labels)
    return {**accuracy_result, **f1_result, **balanced_accuracy_result, **matthews_correlation_result}
```

We also need a mapping between labels and ids in order for the model to perform loss calculation in training. You need to make sure they are correct as it will lead to NaN is loss computation resulting the model laerning nothing at all. 

```python
id2label = {0: "J", 1: "P"}
label2id = {"J": 0, "P": 1}
```

We can now define the model with `AutoModel` from `transformers`. 

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label=id2label, label2id=label2id
)
```

## Fine Tune the model

In order to use provided `AutoModelForSequenceClassification` class, we need to define the training arguments. 

```python 
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

training_args = TrainingArguments(
    output_dir=TRAINED_MODEL,
    learning_rate=2e-5,
    per_device_train_batch_size=4, # modify based on your vRAM
    per_device_eval_batch_size=4, # modify based on your vRAM
    num_train_epochs=1, 
    weight_decay=0.01,
    eval_strategy="steps", # modify based on your data size
    save_strategy="steps", # modify based on your data size
    logging_steps=1000, # modify based on your data size
    eval_steps=1000, # modify based on your data size
    save_steps=1000, # modify based on your data size
    load_best_model_at_end=True,
    push_to_hub=True,
    optim="adamw_bnb_8bit",
    eval_accumulation_steps=2, # modify based on your vRAM
    gradient_accumulation_steps=2, # modify based on your vRAM
    tf32=True,
)
```

In this training argument, several methods are used to speed up the training process and reduce the memory usage. 
- batch size is set to 4 to reduce the memory usage.
- `optim="adamw_bnb_8bit"`: use AdamW with ByteNetBlock 8-bit quantization. This will reduce the vRAM usage by 4 times. You can see more information about this [here](https://huggingface.co/docs/transformers/perf_train_gpu_one#optimizer-choice). 
- `gradient_accumulation_steps=2` and `eval_accumulation_steps=2` to effectively increase the batch size to 8. 
- `tf32=True` to use TensorFloat32 for training.

Here is a comparison between different float types.

![precision-comparison](../assets/img/posts/float_dtype_comparison.png)

With this visualization, you can see that using tf32 will sacrifice precision for less memory usage. TF32 takes 19 bits instead of 32 bits for each float, thus reducing the memory usage by 40%. According to [NVIDIA](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/), research has shown that model trained with tf32 and models trained with fp32 perform very similarly. 

> You can see that we did not use bf16 to fine tune the model. According to the model configuration, Phi3 model is actualy trained with bf16. However, due to some unkown reason, the loss will be NaN if we use bf16 in the fine tuning process. 


Now, we can define the trainer and start training the model. 

```python
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4]).to('cuda'))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mbti_data["train"],
    eval_dataset=tokenized_mbti_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

Here, we wrote a custom trainer with a special loss function. As we discussed before, our data is not balanced with roughly a 60-40 split. Thus, we use a weight balancing method in order to prevent the model turns into a majority calssifier. 

Finnaly, we can actually train the model. 

```python
trainer.train()
trainer.evaluate()
trainer.save_model(TRAINED_MODEL)
trainer.push_to_hub()
```

These code will train the model and push the trained model to your Huggingface repository. 

## Results 

You should be able to see the training process in the console. 
If you increase the training size and evaluation size, you will find the results become more stable and MCC to increase over the time. 