# Large Language Models: A Survey
* Authors: Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, Jianfeng Gao
* Published February 9, 2024

### Paper Structure
![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/f22f3adf-7ad6-4e96-9857-36c85377bda8)

## Overview

*  Language modeling using neural networks was pioneered by [38], [39], [40]. Bengio et al. [13] developed one of the first neural language models (NLMs) that are comparable to n-gram models. Then, [14] successfully applied NLMs to machine translation. The release of RNNLM (an open source NLM toolkit) by Mikolov [41], [42] helped significantly popularize NLMs. Afterwards, NLMs based on recurrent neural networks (RNNs) and their variants, such as long short-term memory (LSTM) [19] and gated recurrent unit (GRU) [20], were widely used for many natural language applications including machine translation, text generation and text classification [43]. Then, the invention of the Transformer architecture [44] marks another milestone in the development of NLMs. By applying self-attention to compute in parallel for every word in a sentence or document an “attention score” to model the influence each word has on another, Transformers allow for much more parallelization than RNNs, which makes it possible to efficiently pre-train very big language models on large amounts of data on GPUs. These pre-trained language models (PLMs) can be fine-tuned for many downstream tasks.

**We group early popular Transformer-based PLMs, based on their neural architectures, into three main categories: encoder only, decoder-only, and encoder-decoder models.**

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/91447624-21ef-4574-a859-a54874f38ae8)


![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/351c355a-d411-452a-9f9a-05449541c192)


![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/1085c9e0-2d9d-4ce8-8374-c8225a577c7f)

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/872f7175-f5ca-43bd-878c-83ae58cc690d)

![image](https://github.com/MuradAladdinzade/LLM_Survey/assets/142248290/ca850417-50f2-4bd3-9b67-797042569fcb)






