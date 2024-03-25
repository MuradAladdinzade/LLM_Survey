# Large Language Models: A Survey
* Authors: Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, Jianfeng Gao
* Published February 9, 2024

### Paper Structure
![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/f22f3adf-7ad6-4e96-9857-36c85377bda8)

## Overview

*  The use of neural networks for language modeling was initially explored in early studies. Bengio and colleagues were among the first to create a neural language model that performed comparably to traditional n-gram models. Following this, neural language models were applied to tasks like machine translation, gaining popularity, especially after Mikolov introduced RNNLM, an open-source toolkit. These models, particularly those using recurrent neural networks (RNNs) and their variations like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), became widely used for various natural language processing tasks, including machine translation, text generation, and classification. The introduction of the Transformer architecture was a significant advancement, utilizing self-attention mechanisms to calculate the relevance of each word to others in a sentence. This allowed for greater parallel processing, enabling the training of very large models on GPUs efficiently. These pre-trained models can be further refined or fine-tuned for specific tasks.

**We group early popular Transformer-based PLMs, based on their neural architectures, into three main categories: encoder only, decoder-only, and encoder-decoder models.**

1. Encoder-only PLMs: As the name suggests, the encoder only models only consist of an encoder network. These models are originally developed for language understanding tasks, such as text classification, where the models need to predict a class label for an input text. Representative encoder-only mod els include BERT and its variants, e.g., RoBERTa, ALBERT, DeBERTa.


2. Decoder-only PLMs: Two of the most widely used decoder-only PLMs are GPT-1 and GPT-2, developed by OpenAI. These models lay the foundation to more powerful LLMs subsequently, i.e., GPT-3 and GPT-4. GPT-1 demonstrates for the first time that good performance over a wide range of natural language tasks can be obtained by Generative Pre-Training (GPT) of a decoder-only Transformer model on a diverse corpus of unlabeled text in a self-supervised learning fashion (i.e., next word/token prediction), followed by discriminative fine-tuning on each specific downstream task (with much fewer samples), as illustrated in Fig 7. GPT-1 paves the way for subsequent GPT models, with each version improving upon the architecture and achieving better performance on various language tasks.
![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/91447624-21ef-4574-a859-a54874f38ae8)

  GPT-2 shows that language models are able to learn to perform specific natural language tasks without any explicit supervision when trained on a large WebText dataset consisting of millions of webpages.    The GPT-2 model follows the model designs of GPT-1 with a few modifications: Layer normal ization is moved to the input of each sub-block, additional layer normalization is added after the final self-      attention block, initialization is modified to account for the accumulation on the residual path and scaling the weights of residual layers, and context size is increased from 512 to 1024 tokens

3. Encoder-Decoder PLMs: Some example PLMs include MASS (MAsked Sequence to Sequence pre-training) and Text-to-Text Transfer Transformer (T5) model. 



### Large Language Model Families

Large language models (LLMs) mainly refer to transformer-based PLMs that contain tens to hundreds of billions of parameters. Compared to PLMs reviewed above, LLMs are not only much larger in model size, but also exhibit stronger language understanding and generation and emergent abilities that are not present in smaller-scale models.

We have couple of LLM families as illustrated by Figure 8. 

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/351c355a-d411-452a-9f9a-05449541c192)


![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/1085c9e0-2d9d-4ce8-8374-c8225a577c7f)

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/872f7175-f5ca-43bd-878c-83ae58cc690d)

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/ca850417-50f2-4bd3-9b67-797042569fcb)






