project was created by:
- Maciej Kasztelanic
- Joanna Jędrzejewska
- Sergio Carcamo

# How to run (step by step)
1. If you want to be able to work with github within container, create .env file by copying .example-env and fill required fields
2. Download model weights, and place them in weights folder. URL: https://drive.google.com/file/d/1jnYn3kaVyoLcGEmDBCOFNPAslrs2tRo4/view
3. build container, it downloads both R and python, and git. If you want to downlaod version with CUDA, swap Dockerfile content with CUDADockerfile
4. run source("project_setup.R") in R console
5. run source("main.R")
6. run in R console run_main(prompt_text = `your prompt`) to run inference test
7. run in R console run_main(mode = "hellaswag") to run hellaswag test

# How to contribute:
- the main training file follows the https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py rewriten in R + reticulate
- to load the proper R libraries and python packages just run source("project_setup.R") which is on top of the model_train.R file
- Current packages in use:
   - R6: most python like OOP in R
   - reticulate: packages that allows running python in R
   - torch for R
   - numpy
   - tiktoken: OpenAI word tokenizer used in GPT2 training process

# RR_project
This project aims to reimplement the GPT-2 model 124M parameters model in R and recreates the zero-shot evaluation of GPT-2 from "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) on three key benchmarks:
   - Language Modeling: Perplexity scores on Penn Treebank (PTB) and WikiText-2 to measure generative performance.
   - Reading Comprehension: Zero-shot F1 score on the CoQA dataset, where the model answers questions conditioned on a document and conversation history.

### Useful links
- Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?fbclid=IwY2xjawJU5gtleHRuA2FlbQIxMAABHSVvjWZf8vA2RR7JqgEaG72UZozbiDUic2HeG25LlcrBUCMIJmgAPyST4w_aem_xSP20pfVnyHvMHvpSq9wgQ
- Video DIY: https://www.youtube.com/watch?v=l8pRSuU81PU
- Python implementation Repo link: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
- Training Data smaller version of webtext https://www.kaggle.com/datasets/isamuisozaki/roughly-one-quarter-of-openwebtext
- Hellaswag ledearboard benchmark: https://github.com/ggml-org/llama.cpp/discussions/2321

## Error messege:
cuda_is_available()
[W524 18:34:28.488601101 CUDAFunctions.cpp:108] Warning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 500: named symbol not found (function operator())
[1] FALSE
Dockerfile with CUDA setup can be viewed in additional_files/CUDADockerfile, but in the end we can run it without GPU usage
