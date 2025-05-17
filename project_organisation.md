# Project Organization
## What needs to be done
1. (Dev) Attention classes (CausalSelfAttention, MLP, Block)
2. (Dev) Model classes (GPTConfig, GPT)
3. (Dev) Data loading classes (DataLoaderLite)
4. (Dev) Launching script
5. (DevOps) Lambda/other GPU provider connection (we can use https://github.com/mdneuzerling/lambdr)
6. (Test) Processing and Testing on COQA
7. (Test) Processing and Testing on PTBdata
8. (Test) Processing and Testing on WikiText2
9. (Presentation) Presentation deadline 24.04.2025

## Team members
1. Sergio Carcamo
2. Joanna Jędrzejewska
3. Maciej Kasztelanic

## Why we chose the subject
We believe the recreation of the revolutionary GPT2 based on the original paper ("Language Models are Unsupervised Multitask Learners")
is gonna be interesting, challanging but its a project everyone could put in their CV. This project can learn us a lot about the basics
of the transformer architecture, help us understand basics of attention mechanism.

## How we plan to solve the problem
We plan to understand the GPT2 architecture by following Andrej Karpathy video about the paper and python implementation of this model. We then want to rewrite it in R language, train it on AWS lamda GPUs and compare our results for chosen datasets with the results authors of GPT2 reported.

## What tools we are going to use
1. R language
2. reticulate library for python interactions
3. R6 for Object Oriented Programming in R
4. Torch for net modelling
5. AWS Lambda for GPU on demand
6. additional libraries
