# NOTE: Add new installs and imports to the project_setup.R file:
source('project_setup.R')
source('data/download_data.R')
download_split("val")

source('data/data_loader_classes.R')
source('model/attention_classes.R')

source('model/gpt_model.R')

source("eval/inference_test.R")
source("eval/hellaswag_test.R")

# Set seeds
torch_manual_seed(1337)
if (cuda_is_available()) {
    torch_cuda_manual_seed(1337)
}

# Initialize tokenizer (using tiktoken via reticulate)
enc <- tiktoken$get_encoding("gpt2")

# -----------------------------------------------------------------------------
# Load Weights and Test Inference

# Define the path to your saved weights file
weights_path <- "weights/Model-weights.pt"

if (!file.exists(weights_path)) {
  stop(sprintf("Weights file not found at: %s", weights_path))
}
message(sprintf("Loading weights from %s", weights_path))

# Load the checkpoint
checkpoint <- torch_load(weights_path)
original_weights <- checkpoint

# Use config from checkpoint if available
if (!is.null(checkpoint$config)) {
  loaded_config <- checkpoint$config
  message("Using model configuration from the loaded checkpoint.")
} else {
  loaded_config <- train_config
  message("Using default training configuration for the model.")
}

# Instantiate model
test_model <- GPT(config = loaded_config)$to(device = loaded_config$device)

# load and process weights
source('model/load_weights.R')

# Set the model to evaluation mode
test_model$eval()

# -----------------------------------------------------------------------------
# Check Inference
# prompt_text <- "Hello, I'm Mike,"
# max_length <- 32 # Maximum length of generated text
# num_return_sequences <- 4 # Number of sequences to generate

# generate_from_prompt(test_model, prompt_text, max_length, loaded_config)

# -----------------------------------------------------------------------------
# Hellaswag Test
hellaswag_test(test_model, enc, loaded_config)