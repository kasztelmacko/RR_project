# RR_project
This project aims to reimplement the GPT-2 model 124M parameters model in R and recreates the zero-shot evaluation of GPT-2 from "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) on three key benchmarks:
   - Language Modeling: Perplexity scores on Penn Treebank (PTB) and WikiText-2 to measure generative performance.
   - Reading Comprehension: Zero-shot F1 score on the CoQA dataset, where the model answers questions conditioned on a document and conversation history.

### Useful links
- Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?fbclid=IwY2xjawJU5gtleHRuA2FlbQIxMAABHSVvjWZf8vA2RR7JqgEaG72UZozbiDUic2HeG25LlcrBUCMIJmgAPyST4w_aem_xSP20pfVnyHvMHvpSq9wgQ
- Video DIY: https://www.youtube.com/watch?v=l8pRSuU81PU
- Python implementation Repo link: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
- Penn Treebank dataset: https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset
- WikiText-2 dataset: https://huggingface.co/datasets/mindchain/wikitext2
- CoQA dataset: https://huggingface.co/datasets/stanfordnlp/coqa
