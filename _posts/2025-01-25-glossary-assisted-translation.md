---
layout: post
title: Use Glossary to Assist AI Translation
subtitle: Translate domain jargons with a simple glossary file.
# cover-img: /assets/img/path.jpg
# thumbnail-img: /assets/img/thumb.png
# share-img: /assets/img/path.jpg
tags: [tutorial, unfinished]
author: Minhao Zhang
---

## Motivation 

This project started off with my wish to translate some videos from English into Chinese.
However, these videos I wish to translate contains a lot of domain-specific words that makes general translation model struggle.

## Why is this even necessary 

Machine Translation has come a long way and it is now possible to translate between many languages with a high degree of accuracy.
However, many current translation methods fail when you are translating a niche topic or your have a lot of jargons in your text. 
Of course, there are services that will allow you to incorperate a glossary file into machine translation like [Google](https://cloud.google.com/translate/docs/advanced/glossary) or [DeepL](https://support.deepl.com/hc/en-us/articles/360021634540-About-the-glossary). 
They are wonderful services that you can use right now to build a robust and reliable translation with glossary support. 

Both Google and DeepL allows user to provide phrases in different languages. 
For instance, if you want to translate `Google Home` as a product name into Chinese, it is likely that it shall remain `Google Home` in the Chinese translation. 
Thus, providing a glossary works pretty well. 
In fact, if the glossary are mostly nouns, this will work almost 100%. 
However, this approach falls short when the jargons are more domain specific and mixed nouns, verbs, and nouns used as verbs. 

Here is a weird exmaple. 
For instance, the word `cat` refers to a small and cute animal that can be kept as pet.
However, if it is spoken by a Linux programmer that only uses CLI and nothing else, it can act like a verb meaning reading files sequentially and write them into standard output. 

Here is another example.
> He pillars up with dirt and gravel. 

This might looks like gibberish to most people. 
However, it makes perfect sense if you are playing Minecraft. 
In this case, the word "pillar" referes to the action that player repeatedly places a block underneath the player while jumping in the air in order to go to high places. 
Here, `pillar`, normally a noun, transformed into a verb in a linguistic process called verbing or denominal. 
The current commercially available approach is not really suited for this task. 
Thus, I decided to write one myself. 


## What enables this project 

Recently, LLMs has improved to a point that locally run models are performing extremely well. 
These models, though lacking complex reasoning abilities, can follow accurate instructions and produce coherent results in both general chat and translation tasks. 
With tools like `Ollama` and `vllm`, you can run many quantized LLM on your laptop. 
On my RTX 4070 laptop with 8GB VRAM, I can comfortablly run `qwen2.5:7b` with `Q4_K_M` quantization using Ollama.

With locally run LLMs, we can then use Zero-Shot or Chain-of-Thoughts prompting to provide relavent information to LLM so it can understand the context better. 
Different from a raw glossary look-up table, we can provide definitons and examples to the LLM so it can understand the glossary better. 


