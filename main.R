run_main <- function(mode = "inference", prompt_text = "Default Prompt for you,") {
#Global flags (for caching within the session)
  if (!exists("SETUP_DONE")) SETUP_DONE <<- FALSE
  if (!exists("WEIGHTS_LOADED")) WEIGHTS_LOADED <<- FALSE
  if (!exists("MODEL_INITIALIZED")) MODEL_INITIALIZED <<- FALSE
#Project settup
  if (!SETUP_DONE) {
    message("Running project setup...")
    source('project_setup.R')
    source('data/download_data.R')
    source('data/data_loader_classes.R')
    download_split("val")
    SETUP_DONE <<- TRUE
  } else {
    message("Project setup already done — using cached setup.")
  }

  torch_manual_seed(1337)
  if (cuda_is_available()) {
    torch_cuda_manual_seed(1337)
  }

  message("Initializing tokenizer...")
  enc <<- tiktoken$get_encoding("gpt2")

  weights_path <- "weights/Model-weights.pt"
  if (!WEIGHTS_LOADED) {
    if (!file.exists(weights_path)) {
      stop(sprintf("Weights file not found at: %s", weights_path))
    }
    message(sprintf("Loading weights from %s", weights_path))
    checkpoint <- torch_load(weights_path)
    original_weights <<- checkpoint
    WEIGHTS_LOADED <<- TRUE
  } else {
    message("Model weights already loaded — using cached weights.")
  }

  source('model/attention_classes.R')
  source('model/gpt_model.R')
  if (!is.null(original_weights$config)) {
    loaded_config <- original_weights$config
    message("Using model configuration from the loaded checkpoint.")
  } else {
    if (!exists("train_config")) stop("No config found in checkpoint and 'train_config' not defined.")
    loaded_config <- train_config
    message("Using default training configuration for the model.")
  }
#Initialize model (only once per session)
  if (!MODEL_INITIALIZED) {
    message("Initializing model...")
    test_model <<- GPT(config = loaded_config)$to(device = loaded_config$device)
    source('model/load_weights.R')
    test_model$eval()
    MODEL_INITIALIZED <<- TRUE
  } else {
    message("Model already initialized — using cached instance.")
  }
#Run specified test
  if (mode == "inference") {
    message("Running inference test...")
    source("eval/inference_test.R")
    generate_from_prompt(test_model, prompt_text, max_length = 32, loaded_config)
  } else if (mode == "hellaswag") {
    message("Running Hellaswag evaluation...")
    source("eval/hellaswag_test.R")
    hellaswag_test(test_model, enc, loaded_config)
  } else {
    stop(paste("Unknown mode:", mode, "\nAvailable modes: inference, hellaswag"))
  }
}
