# Large Language Models: A Survey
* Authors: Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, Jianfeng Gao
* Published February 9, 2024

### Paper Structure
![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/f22f3adf-7ad6-4e96-9857-36c85377bda8)

## Overview

*  The use of neural networks for language modeling was initially explored in early studies. Bengio and colleagues were among the first to create a neural language model that performed comparably to traditional n-gram models. Following this, neural language models were applied to tasks like machine translation, gaining popularity, especially after Mikolov introduced RNNLM, an open-source toolkit. These models, particularly those using recurrent neural networks (RNNs) and their variations like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), became widely used for various natural language processing tasks, including machine translation, text generation, and classification. The introduction of the Transformer architecture was a significant advancement, utilizing self-attention mechanisms to calculate the relevance of each word to others in a sentence. This allowed for greater parallel processing, enabling the training of very large models on GPUs efficiently. These pre-trained models can be further refined or fine-tuned for specific tasks.

**We group early popular Transformer-based PLMs, based on their neural architectures, into three main categories: encoder only, decoder-only, and encoder-decoder models.**

1. Encoder-only PLMs: As the name suggests, the encoder only models only consist of an encoder network. These models are originally developed for language understanding tasks, such as text classification, where the models need to predict a class label for an input text. Representative encoder-only mod els include BERT and its variants, e.g., RoBERTa, ALBERT, DeBERTa, XLM, XLNet, UNILM, as to be described below.

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/91447624-21ef-4574-a859-a54874f38ae8)


![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/351c355a-d411-452a-9f9a-05449541c192)


![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/1085c9e0-2d9d-4ce8-8374-c8225a577c7f)

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/872f7175-f5ca-43bd-878c-83ae58cc690d)

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/ca850417-50f2-4bd3-9b67-797042569fcb)






