---
title: Fine Tune Adapter Layer for Embedding Models
categories:
  - blog
tags:
  - en
  - tutorial
# toc: True
classes: wide

---

## Introduction

Fine-tuning has never been easier. With the advent of LoRA (Low Rank Adaptation) and its quantized counterpart—supported by frameworks such as [unsloth](https://github.com/unslothai/unsloth) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)—anyone can fine-tune nearly any LLM using affordable computing resources. These models, known for their robust foundations and flexible adaptation capabilities, have captured the attention of the entire LLM community. However, when constructing a RAG (Retrieval Augmented Generation) system, the quality of the embedding model is equally critical.

A RAG system acts as a private knowledge base that allows for rapid retrieval of relevant documents. High-quality embeddings ensure that retrieved documents accurately reflect the nuances of your domain-specific content. General embedding models—trained on vast amounts of diverse text—may underperform in specialized fields where terminology and usage deviate from the norm. By fine-tuning an embedding model to better align with your specific domain, you can significantly improve retrieval accuracy.

Moreover, this method offers a substantial speed advantage over traditional large embedding models. By focusing on the adapter layer, which only requires modifying a small fraction of the overall parameters, the fine-tuning process becomes incredibly fast. This efficiency not only reduces the computational cost but also accelerates the adaptation cycle, allowing for rapid deployment and iteration in dynamic environments.

## Theory

Proposed by Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) has become the de facto method for fine-tuning LLMs. Instead of training every parameter in a model—which can number in the billions—LoRA injects trainable rank decomposition matrices into each layer of the Transformer architecture. This approach reduces the number of trainable parameters by up to 10,000 times while significantly lowering memory requirements.

Similarly, fine-tuning an entire embedding model can be resource-intensive. To address this challenge, a comparable strategy can be applied by inserting an adapter layer immediately after the embedding process. This method drastically cuts down the number of parameters that need training, leading to faster training times and reduced computational overhead.

Furthermore, by training an adapter exclusively for query embeddings, we align them with the pre-existing embedding space without the need to recalculate document embeddings. In practice, each query is first embedded using the existing model, then refined via the adapter layer. This targeted approach not only accelerates the fine-tuning process but also preserves the integrity of the document embeddings, resulting in a swift and efficient system.

## Trainable Layer

As demonstrated in rigorous testing by the research team at Chroma [here](https://research.trychroma.com/embedding-adapters):

> The simplest such adapter is a linear transform, which can be implemented efficiently as a matrix multiplication. In this work we find that training an adapter applied to **just the query embedding**, from relatively few labeled query-document pairs (as few as 1,500), produces an improvement in retrieval accuracy over the pre-trained embedding model alone of **up to 70%**. These labels can be collected from human feedback, for example from application usage. Further, we show that negative examples generated as random samples from documents not labeled as relevant to a particular query, is sufficient to achieve this improvement, considerably reducing labeling cost.

In this work, I employ the same strategy proposed by the Chroma research team, with an additional twist: an MLP (Multi-layer Perceptron) adapter featuring a single hidden layer of 2048 dimensions. This extra layer further refines the query embeddings, enhancing the performance without compromising speed.

### Example Models and Their Trainable Parameters

| Model             | Model Params | Embed Dim | Linear Adapter Params | MLP Adapter Params    |
| ----------------- | ------------ | --------- | --------------------- | --------------------- |
| all-MiniLM-L6-v2  | 22.7M        | 384       | 384 x 384 ≈147K       | 384 × 2048 × 2 ≈ 1.6M |
| all-mpnet-base-v2 | 109M         | 768       | 768 x 768 ≈ 590K      | 768 × 2048 × 2 ≈ 3.1M |

The efficiency of this approach is particularly notable. Unlike larger embedding models that require extensive retraining and substantial computational resources, our adapter-based method enables rapid fine-tuning with a minimal parameter footprint. This results in significant speed improvements, making it an ideal solution for applications where quick adaptation and deployment are critical.

## Training 

We can employ a simple way to train this by using a chunk of text as positive, a question that can be answered with that chunk of text, and some random negative that is just irrelevant to our context. After that, we can use a triplet loss to perform the actual training. 

### Data Preparation

Before we start, it is obvious that we need some data to train and test this. I am sure you have some pre-chunked text lying around so just grab those. Now, we need some questions. Ideally, this would be created by human while they are using the RAG system or crafted in advance, but we could always use some synthetic data when we just started exploring. Using the following prompt template, you can ask any LLM to generate some questions for you. 
```text
You are an AI assistant tasked with generating a single, realistic question-answer pair based on a given document. The question should be something a user might naturally ask when seeking information contained in the document.

Given: {chunk}

Instructions:
1. Analyze the key topics, facts, and concepts in the given document, choose one to focus on.
2. Generate ten similar questions that a user might ask to find the information in this document.
3. Use natural language and occasionally include typos or colloquialisms to mimic real user behavior in the question.
4. Ensure the question is semantically related to the document content WITHOUT directly copying phrases.
5. Make sure that all of the questions can be answered by the given document.

Output Format:
Return a JSON object with the following structure:
```json
{
  "question_1": "Generated question text",
  "question_2": "Generated question text",
  ...
}

Be creative, think like a curious user, and generate your 10 questions that would naturally lead to the given document in a semantic search. Ensure your response is a valid JSON object containing only the questions.
```
The above prompt is adapted from [Adam Lucek's notebook](https://github.com/ALucek/linear-adapter-embedding/blob/main/Linear_Adapter.ipynb) 

Now we have the positives and the questions, we need the negatives. This is pretty simple as it can be any text that is irrelevant. I used some books from the [Project Gutenberg](https://www.gutenberg.org/) and chunked them using `langchain_text_splitters.RecursiveCharacterTextSplitter` into roughly the size of our . 

### Precomputed Triplet Dataset
If you are paying attention closely, you will realize that all we are training is the adapter layer. The document embeddings and the query embeddings were never changed during the training. Thus, it is much more efficient to compute all the embedding before we started training. We can define a `torch.Dataset` like this so we get a random negative sample whenever it is called.
```python
class PrecomputedTripletDataset(Dataset):
    def __init__(
        self,
        query_embs: torch.Tensor,
        positive_embs: torch.Tensor,
        negative_embs: torch.Tensor,
    ):
        self.query_embs = query_embs
        self.positive_embs = positive_embs
        self.negative_embs = negative_embs

    def __len__(self) -> int:
        return len(self.query_embs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Randomly select a negative sample
        neg_idx = random.randint(0, len(self.negative_embs) - 1)
        return (
            self.query_embs[idx],
            self.positive_embs[idx],
            self.negative_embs[neg_idx],
        )
```

### Training Adapter 
The actual training of the adapter is nothing out of ordinary. It follows the most usual deep learning guideline except the data is now triplet and the loss is `torch.nn.TripletMarginLoss`. The core training logic is shown below.
```python
dataloader = DataLoader(
	self.dataset, batch_size=batch_size, shuffle=True)
optimizer = AdamW(self.adapter.parameters(), lr=learning_rate)
triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
	optimizer, warmup_steps, total_steps)

epoch_metrics = []  # List to store metrics for each epoch
for epoch in range(num_epochs):
	self.adapter.train()
	
	# Iterate over the batches in the DataLoader
	for batch in dataloader:
		# Move the batch tensors to the device
		q, p, n = [t.to(self.device, non_blocking=True) for t in batch]
		# Pass the query embeddings through the adapter
		adapted_q = self.adapter(q)
		# Calculate the triplet loss
		loss = triplet_loss(adapted_q, p, n)

		# Zero the gradients, perform backpropagation, and update the parameters
		optimizer.zero_grad()
		loss.backward()
		clip_grad_norm_(self.adapter.parameters(), max_grad_norm)
		optimizer.step()
		scheduler.step()
```

Of course these are only part of the training script, you can find the full training logic in [adapter_trainer.py](https://github.com/Minhao-Zhang/Fine-Tune-Embedding-Adapter/blob/main/adapter_trainer.py).  

### Evaluation 
There's a lot of different metrics used for evaluating information retrieval, it is very hard to choose between them. Thus, I included all of them. 
```python

def reciprocal_rank(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    try:
        rank = retrieved_docs.index(ground_truth) + 1
        return 1.0 / rank if rank <= k else 0.0
    except ValueError:
        return 0.0


def hit_rate(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    return 1.0 if ground_truth in retrieved_docs[:k] else 0.0


def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    retrieved_at_k = retrieved_docs[:k]
    return sum(1 for doc in retrieved_at_k if doc in relevant_docs) / k


def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    return sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs) / len(relevant_docs)


def average_precision(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    score, num_hits = 0.0, 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant_docs) if relevant_docs else 0.0


def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    def dcg(scores):
        return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores))

    ideal_scores = [1] * min(k, len(relevant_docs)) + \
        [0] * (k - min(k, len(relevant_docs)))
    actual_scores = [
        1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]

    return dcg(actual_scores) / dcg(ideal_scores) if dcg(ideal_scores) > 0 else 0.0
```

## Results 


## Reference 
- [linear-adapter-embedding by Adam Łucek](https://github.com/ALucek/linear-adapter-embedding)
- [Embedding Adapters by Chroma](https://research.trychroma.com/embedding-adapters)
- [Fine-tune Embedding models for Retrieval Augmented Generation (RAG) by Philipp Schmid](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
