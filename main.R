# -----------------------------------------------------------------------------
# Global flags (for caching within the session)
if (!exists("SETUP_DONE")) SETUP_DONE <- FALSE
if (!exists("WEIGHTS_LOADED")) WEIGHTS_LOADED <- FALSE
if (!exists("MODEL_INITIALIZED")) MODEL_INITIALIZED <- FALSE

# -----------------------------------------------------------------------------
# Project Setup (Install R packages, Python packages, download data)
if (!SETUP_DONE) {
  message("Running project setup...")
  source('project_setup.R')
  source('data/download_data.R')
  source('data/data_loader_classes.R')
  download_split("val")
  SETUP_DONE <- TRUE
} else {
  message("Project setup already done — using cached setup.")
}

# -----------------------------------------------------------------------------
# Set random seeds
torch_manual_seed(1337)
if (cuda_is_available()) {
  torch_cuda_manual_seed(1337)
}

# -----------------------------------------------------------------------------
# Initialize tokenizer
message("Initializing tokenizer...")
enc <- tiktoken$get_encoding("gpt2")

# -----------------------------------------------------------------------------
# Load model weights
weights_path <- "weights/Model-weights.pt"
if (!WEIGHTS_LOADED) {
  if (!file.exists(weights_path)) {
    stop(sprintf("Weights file not found at: %s", weights_path))
  }
  message(sprintf("Loading weights from %s", weights_path))
  checkpoint <- torch_load(weights_path)
  original_weights <- checkpoint
  WEIGHTS_LOADED <- TRUE
} else {
  message("Model weights already loaded — using cached weights.")
}

# -----------------------------------------------------------------------------
# Load model config
if (!is.null(checkpoint$config)) {
  loaded_config <- checkpoint$config
  message("Using model configuration from the loaded checkpoint.")
} else {
  if (!exists("train_config")) stop("No config found in checkpoint and 'train_config' not defined.")
  loaded_config <- train_config
  message("Using default training configuration for the model.")
}

# -----------------------------------------------------------------------------
# Initialize model (only once per session)
if (!MODEL_INITIALIZED) {
  message("Initializing model...")
  source('model/attention_classes.R')
  source('model/gpt_model.R')
  test_model <- GPT(config = loaded_config)$to(device = loaded_config$device)

  source('model/load_weights.R')
  test_model$eval()
  MODEL_INITIALIZED <- TRUE
} else {
  message("Model already initialized — using cached instance.")
}

# -----------------------------------------------------------------------------
# Run test inference
source("eval/inference_test.R")
prompt_text <- "Hello, I'm Mike,"
generate_from_prompt(test_model, prompt_text, max_length = 32, loaded_config)

# -----------------------------------------------------------------------------
# Run Hellaswag Evaluation
# message("Running Hellaswag evaluation...")
# source("eval/hellaswag_test.R")
# hellaswag_test(test_model, enc, loaded_config)
