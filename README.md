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

Some LLM families as illustrated below:

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/351c355a-d411-452a-9f9a-05449541c192)



### HOW LLMS ARE BUILT

In this section, we first review the popular architectures used for LLMs, and then discuss data and modeling techniques ranging from data preparation, tokenization, to pre-training, instruction tuning, and alignment. Once the model architecture is chosen, the major steps involved in training an LLM includes: data preparation (col lection, cleaning, deduping, etc.), tokenization, model pre training (in a self-supervised learning fashion), instruction tuning, and alignment.


_Dominant LLM architectures are Encoder, Encoder-Decoder, and Decoder._


![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/ca850417-50f2-4bd3-9b67-797042569fcb)

####  Data Cleaning
Data quality is crucial to the performance of language models trained on them. Data cleaning techniques such as filtering, deduplication, are shown to have a big impact on
the model performance.
  1. Data Filtering: Data filtering aims to enhance the quality of training data and the effectiveness of the trained LLMs.
  Common data filtering techniques include:
      * Removing Noise: refers to eliminating irrelevant or noisy data that might impact the model’s ability to generalize well.
        As an example, one can think of removing false information from the training data, to lower the chance of model generating
        false responses. Two mainstream approaches for quality filtering includes: classifier-based, and heuristic-based frameworks.
      *  Handling Outliers: Identifying and handling outliers or anomalies in the data to prevent them from disproportionately influencing the model.
      *  Addressing Imbalances: Balancing the distribution of classes or categories in the dataset to avoid biases and ensurefair representation. This is specially useful for responsible
         model training and evaluation.
      *  Text Preprocessing: Cleaning and standardizing text data by removing stop words, punctuation, or other elements that may not contribute significantly to the model’s learning.
      *  Dealing with Ambiguities: Resolving or excluding ambiguous or contradictory data that might confuse the model during training. This can help the model to provide more definite and reliable answers.
 

####  Tokenizations

Tokenization referes to the process of converting a sequence of text into smaller parts, known as tokens. While
the simplest tokenization tool simply chops text into tokens
based on white space, most tokenization tools rely on a word
dictionary. However, out-of-vocabulary (OOV) is a problem
in this case because the tokenizer only knows words in its
dictionary. To increase the coverage of dictionaries, popular
tokenizers used for LLMs are based on sub-words, which can
be combined to form a large number of words, including the
words unseen in training data or words in different languages.
In what follows, we describe three popular tokenizers.
1) BytePairEncoding: BytePairEncoding is originally a
type of data compression algorithm that uses frequent patterns
at byte level to compress the data. By definition, this algorithm
mainly tries to keep the frequent words in their original form
and break down ones that are not common. This simple
paradigm keeps the vocabulary not very large, but also good
enough to represent common words at the same time. Also
morphological forms of the frequent words can be represented
very well if suffix or prefix is also commonly presented in the
training data of the algorithm.
2) WordPieceEncoding: This algorithm is mainly used for
very well-known models such as BERT and Electra. At the
beginning of training, the algorithm takes all the alphabet from
the training data to make sure that nothing will be left as UNK
or unknown from the training dataset. This case happens when
the model is given an input that can not be tokenized by the
tokenizer. It mostly happens in cases where some characters are
not tokenizable by it. Similar to BytePairEncoding, it tries to
maximize the likelihood of putting all the tokens in vocabulary
based on their frequency.
3) SentencePieceEncoding: Although both tokenizers described before are strong and have many advantages compared
to white-space tokenization, they still take assumption of
words being always separated by white-space as granted. This
assumption is not always true, in fact in some languages, words
can be corrupted by many noisy elements such as unwanted
spaces or even invented words. SentencePieceEncoding tries
to address this issue.



