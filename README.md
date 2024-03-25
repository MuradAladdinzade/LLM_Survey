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


## Architeture Overview
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

Tokenization referes to the process of converting a sequence of text into smaller parts, known as tokens. While
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
3) SentencePieceEncoding: Although both tokenizers described before are strong and have many advantages compared
to white-space tokenization, they still take assumption of
words being always separated by white-space as granted. This
assumption is not always true, in fact in some languages, words
can be corrupted by many noisy elements such as unwanted
spaces or even invented words. SentencePieceEncoding tries
to address this issue.


####  Positional Encoding

1) Absolute Positional Embeddings: (APE) has been
used in the original Transformer model to preserve the information of sequence order. Therefore, the positional information
of words is added to the input embeddings at the bottom of
both the encoder and decoder stacks. There are various options
for positional encodings, either learned or fixed. In the vanilla
Transformer, sine and cosine functions are employed for this
purpose. The main drawback of using APE in Transformers
is the restriction to a certain number of tokens. Additionally,
APE fails to account for the relative distances between tokens.

2) Relative Positional Embeddings: (RPE) involves
extending self-attention to take into account the pairwise links
between input elements. RPE is added to the model at two
levels: first as an additional component to the keys, and
subsequently as a sub-component of the values matrix. This
approach looks at the input as a fully-connected graph with
labels and directed edges. In the case of linear sequences, edges
can capture information about the relative position differences
between input elements. A clipping distance, represented as k
2 ≤ k ≤ n − 4, specifies the maximum limit on relative locations. This allows the model to make reasonable predictions
for sequence lengths that are not part of the training data.

3) Rotary Position Embeddings: Rotary Positional Embedding (RoPE) tackles problems with existing approaches. Learned absolute positional encodings can lack generalizability and meaningfulness, particularly when sentences
are short. Moreover, current methods like T5’s positional
embedding face challenges with constructing a full attention
matrix between positions. RoPE uses a rotation matrix to
encode the absolute position of words and simultaneously includes explicit relative position details in self-attention. RoPE
brings useful features like flexibility with sentence lengths, a
decrease in word dependency as relative distances increase,
and the ability to improve linear self-attention with relative
position encoding. GPT-NeoX-20B, PaLM, CODEGEN, and
LLaMA are among models that take advantage of RoPE in
their architectures.
4) Relative Positional Bias: The concept behind this type
of positional embedding is to facilitate extrapolation during
inference for sequences longer than those encountered in training. In Press et al. proposed Attention with Linear Biases
(ALiBi). Instead of simply adding positional embeddings to
word embeddings, they introduced a bias to the attention scores
of query-key pairs, imposing a penalty proportional to their
distance. In the BLOOM model, ALiBi is leveraged.

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/47ffb7b1-f276-4411-8d74-510584763042)

####  Model Pre-training

Pre-training is the very first step in large language model
training pipeline, and it helps LLMs to acquire fundamental
language understanding capabilities, which can be useful in a
wide range of language related tasks. During pre-training, the
LLM is trained on a massive amount of (usually) unlabeled
texts, usually in a self-supervised manner. There are different
approaches used for pre-training like next sentence prediction
[24], two most common ones include, next token prediction
(autoregressive language modeling), and masked language
modeling.
*  In Autoregressive Language Modeling framework, given
a sequence of n tokens x1, ..., xn, the model tries to predict
next token xn+1 (and sometimes next sequence of tokens) in
an auto-regressive fashion. 
*  In Masked Language Modeling, some words are masked
in a sequence and the model is trained to predict the masked
words based on the surrounding context. Sometimes people
refer to this approach as denoising autoencoding, too.


####  Fine-tuning and Instruction Tuning
Early language models such as BERT trained using self-supervision were not able to
perform specific tasks. In order for the foundation model to be
useful it needed to be fine-tuned to a specific task with labeled
data. For
example, in the original BERT paper, the model was fine-tuned to 11 different tasks. While more recent LLMs no longer
require fine-tuning to be used, they can still benefit from task
or data-specific fine-tuning. 

#### Alignment
AI Alignment is the process of steering AI systems towards
human goals, preferences, and principles. LLMs, pre-trained
for word prediction, often exhibit unintended behaviors. For
example, they might generate contents that are toxic, harmful,
misleading and biased.
Instruction tuning, discussed above, gets LLMs a step
closer to being aligned. However, in many cases, it is important
to include further steps to improve the alignment of the model
and avoid unintended behaviors.


#### Decoding Strategies
Decoding refers to the process of text generation using pretrained LLMs. Given an input prompt, the tokenizer translates
each token in the input text into a corresponding token ID.
Then, the language model uses these token IDs as input and
predicts the next most likely token (or a sequence of tokens).
Finally, the model generates logits, which are converted to
probabilities using a softmax function. Different decoding
strategies have been proposed. Some of the most popular ones
are greedy search, beam search, as well as different sample
techniques such as top-K.

1) Greedy Search: Greedy search takes the most probable
token at each step as the next token in the sequence, discarding
all other potential options. As you can imagine, this is a simple
approach and can loose a lot of temporal consistency and
coherency. It only considers the most probable token at each
step, without considering the overall effect on the sequence.

2) Beam Search: Unlike greedy search that only considers
the next most probable token, beam search takes into account
the N most likely tokens, where N denotes the number of
beams. This procedure is repeated until a predefined maximum sequence length is reached or an end-of-sequence token
appears. At this point, the sequence of tokens (AKA “beam”)
with the highest overall score is chosen as the output. For
example for beam size of 2 and maximum length of 5,
the beam search needs to keep track of 2^5 = 32 possible
sequences. So it is more computationally intensive than greedy
search.

3) Top-k Sampling: Top-k sampling is a technique that
uses the probability distribution generated by the language
model to select a token randomly from the k most likely
options.
Suppose we have 6 tokens (A, B, C, D, E, F) and k=2,
and P(A)= 30%, and P(B)= 20%, P(C)= P(D)= P(E)= P(F)=
12.5%. In top-k sampling, tokens C, D, E, F are disregarded,
and the model outputs A 60% of the time, and B, 40% of
the time. This approach ensures that we prioritize the most
probable tokens while introducing an element of randomness
in the selection process.

 #### Cost-Effective Training/Inference/Adaptation/Compression

Let's review some of the popular approaches
used for more cost-friendly (and compute-friendly) training
and usage of LLMs.

1) Low-Rank Adaption (LoRA): Low-Rank Adaptation is
a popular and lightweight training technique that significantly
reduces the number of trainable parameters, and is based
on a crucial insight that the difference between the fine-tuned weights for a specialized task and the initial pre-trained
weights often exhibits “low intrinsic rank” - meaning that
it can be approximated well by a low rank matrix.

**Key Differences and Advantages of LORA**
Training with LoRA is much faster, memory-efficient, and
produces smaller model weights (a few hundred MBs), that are
easier to store and share. One property of low-rank matrices
is that they can be represented as the product of two smaller
matrices. This realization leads to the hypothesis that this delta
between fine-tuned weights and initial pre-trained weights can
be represented as the matrix product of two much smaller
matrices. By focusing on updating these two smaller matrices
rather than the entire original weight matrix, computational
efficiency can be substantially improved.

**Formal Pseudocode for LORA**

- **Input:**
  - Initial weight matrix: <code>W<sub>0</sub> ∈ ℝ<sup>d × k</sup></code>
  - Vector for input: <code>x ∈ ℝ<sup>d</sup></code>
- **Output:**
  - Modified output vector: <code>h ∈ ℝ<sup>k</sup></code>
- **Hyperparameters:**
  - Low rank: <code>r ∈ ℕ, r &ll; min(d, k)</code>
  - Adjustment coefficient: <code>α ∈ ℝ</code>
- **Adjustment Parameters:**
  - Decomposed matrices: <code>A ∈ ℝ<sup>r × k</sup>, B ∈ ℝ<sup>d × r</sup></code>

1. Begin with zeroed update matrix: <code>ΔW &larr; 0<sub>d × k</sub></code>
2. Iterate from <code>i = 1</code> to <code>r</code>:
   - Iterate from <code>j = 1</code> to <code>k</code>:
     - Modify <code>ΔW<sub>:, j</sub> &larr; ΔW<sub>:, j</sub> + B<sub>:, i</sub> ⋅ A<sub>i, j</sub></code>
3. Refresh weight matrix: <code>W &larr; W<sub>0</sub> + α/r ⋅ ΔW</code>
4. Determine final output: <code>h &larr; W ⋅ x</code>
5. **Return** the modified output vector `h`
   
Specifically, for a pre-trained weight matrix W0 ∈ Rd×k
,
LoRA constrains its update by representing the latter with
a low-rank decomposition W0 + ∆W = W0 + BA, where
B ∈ Rd×r
, A ∈ Rr×k
, and the rank r ≪ min(d, k). During
training, W0 is frozen and does not receive gradient updates,
while A and B contain trainable parameters. It is worth
mentioning that both W0 and ∆W = BA are multiplied with
the same input, and their respective output vectors are summed
coordinate-wise. For h = W0x, their modified forward pass
yields: h = W0x + ∆W x = W0x + BAx. Usually a random
Gaussian initialization is used for A, and zero initialization
for B, so ∆W = BA is zero at the beginning of training.
They then scale ∆W x by αr, where α is a constant in r. This
reparametrization is illustrated below:
![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/d573a123-b4d1-4fb5-8dbc-abb834db453f)

2) Knowledge Distillation: Knowledge distillation is the
process of learning from a larger model. Earlier days of
best-performing models release have proven that this approach
is very useful even if it is used in an API distillation approach.

**Question 1**
* Is it possible to distill the knowledge of
multiple models into a smaller
one?
<details>
  <summary>:Click to Show Answer</summary>
Yes, it is possible.
</details>
  

Creating smaller models by this approach yields smaller
model sizes that can be used even on edge devices. Knowledge
distillation as shown in Fig 35, illustrates a general setup of
this training scheme.

Knowledge can be transferred by different forms of learning: response distillation, feature distillation, and API distillation. Response distillation is concerned only with the outputs
of the teacher model and tries to teach the student model
how to exactly or at least similarly perform (in the sense of
prediction) as the teacher. Feature distillation not only uses
the last layer but also intermediate layers as well to create a
better inner representation for the student model. This helps the
smaller model to have a similar representation as the teacher
model.

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/e9e1d276-1152-49cc-a7d0-69613c850810)

## Critical Analysis
**What was overlooked by the authors?**
* One of the drawbacks of the paper is its failure to detail the technical aspects of Knowledge Distillation. While it highlights the method's usefulness in creating smaller, efficient models suitable for edge devices, it lacks a deep dive into the process's technical workings. The author should have included a detailed explanation of how Knowledge Distillation operates from a technical perspective.

**What could have been developed further?**
* Although main models were compared using different benchmarks, variants/fine-tuned models could have also be compared with each other, and results could have be reported to illustrate better fine-tuning processes. 




## Resource links

* LORA from Scratch: https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
* Tokenization: https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html
* A Comprehensive Overview of Large Language Models: https://arxiv.org/abs/2307.06435
* Gihub repo for the book: Build a Large Language Model: https://github.com/rasbt/LLMs-from-scratch
* Impact of Decoding Strategies on Instruction Following: https://deci.ai/blog/llm-evaluation-and-how-decoding-strategies-impact-instruction-following/

## Paper Citation
Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher, R., Amatriain, X., & Gao, J. (2024). Large language models: A survey. arXiv preprint arXiv:2402.06196. Retrieved from https://arxiv.org/abs/2402.06196
