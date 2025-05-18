# NOTE: Add new installs and imports to the project_setup.R file:
source('project_setup.R')
source('data_loader_classes.R')
source('attention_classes.R')

# -----------------------------------------------------------------------------
# GPTConfig (Using an R list as started by the user)
# Add dropout parameters to the config list
GPTConfig <- list(
  block_size = 1024L, # max sequence length
  vocab_size = 50257L, # for model training is 50304L according to Python comments
  n_layer = 12L, # number of layers
  n_head = 12L,  # number of heads
  n_embd = 768L, # embedding dimension
  attn_dropout = 0.0, # Example dropout values, adjust as needed
  resid_dropout = 0.0
)

# -----------------------------------------------------------------------------
# GPT Model (translation of Python's GPT class)
GPT <- nn_module(
  "GPT",
  initialize = function(config) {
    super$initialize()
    self$config <- config

    # Using nn_module_dict for the transformer components
    self$transformer <- nn_module_dict(list(
      wte = nn_embedding(config$vocab_size, config$n_embd),
      wpe = nn_embedding(config$block_size, config$n_embd),
      h = nn_module_list(lapply(1:config$n_layer, function(i) Block(config))),
      ln_f = nn_layer_norm(config$n_embd)
    ))
    self$lm_head <- nn_linear(config$n_embd, config$vocab_size, bias = FALSE)

    # weight sharing scheme
    # self$transformer[["wte"]]$weight <- self$lm_head$weight

    # init params
    self$apply(self$.init_weights)
  },

  # Helper function for weight initialization
  .init_weights = function(module) {
    if (inherits(module, "nn_linear")) {
      std <- 0.02
      if (!is.null(attr(module, "NANOGPT_SCALE_INIT"))) {
        std <- std * (2 * self$config$n_layer)^-0.5
      }
      nn_init_normal_(module$weight, mean = 0.0, std = std)
      if (!is.null(module$bias)) {
        nn_init_zeros_(module$bias)
      }
    } else if (inherits(module, "nn_embedding")) {
      nn_init_normal_(module$weight, mean = 0.0, std = 0.02)
    }
  },

  forward = function(idx, targets = NULL) {
    Batch <- idx$size(1)
    Token <- idx$size(2)
    if (Token > self$config$block_size) {
      stop(sprintf("Cannot forward sequence of length %d, block size is only %d", 
                  Token, self$config$block_size))
    }

    # Create positions starting from 1 for R's 1-based indexing
    # torch_arange(1, Token+1) creates indices [1, 2, ..., Token]
    pos <- torch_arange(1, Token + 1, dtype = torch_long(), device = idx$device)

    pos_emb <- self$transformer[["wpe"]](pos)
    tok_emb <- self$transformer[["wte"]](idx)

    x <- tok_emb + pos_emb

    # Alternative iteration method for transformer blocks
    h_blocks <- self$transformer[["h"]]
    for (i in seq_along(h_blocks)) {
      x <- h_blocks[[i]](x)
    }

    x <- self$transformer[["ln_f"]](x)
    logits <- self$lm_head(x)

    loss <- NULL
    if (!is.null(targets)) {
      loss <- nnf_cross_entropy(logits$view(c(-1, logits$size(-1))), targets$view(-1))
    }

    return(list(logits = logits, loss = loss))
  },

  from_pretrained = function(model_type) {
    stop("from_pretrained is not fully implemented in this R version.")
  },

  configure_optimizers = function(weight_decay, learning_rate, device_type, master_process = TRUE) {
    param_dict <- self$named_parameters()
    param_dict <- param_dict[sapply(param_dict, function(p) p$requires_grad)]

    decay_params <- list()
    nodecay_params <- list()

    for (name in names(param_dict)) {
      p <- param_dict[[name]]
      if (p$dim() >= 2) {
        decay_params[[name]] <- p
      } else {
        nodecay_params[[name]] <- p
      }
    }

    optim_groups <- list(
      list('params' = decay_params, 'weight_decay' = weight_decay),
      list('params' = nodecay_params, 'weight_decay' = 0.0)
    )

    num_decay_params <- sum(sapply(decay_params, function(p) p$numel()))
    num_nodecay_params <- sum(sapply(nodecay_params, function(p) p$numel()))

    if (master_process) {
      message(sprintf("num decayed parameter tensors: %d, with %s parameters", length(decay_params), format(num_decay_params, big.mark = ",")))
      message(sprintf("num non-decayed parameter tensors: %d, with %s parameters", length(nodecay_params), format(num_nodecay_params, big.mark = ",")))
    }

    optimizer <- optim_adamw(optim_groups, lr = learning_rate, betas = c(0.9, 0.95), eps = 1e-8)
    return(optimizer)
  }
)

# -----------------------------------------------------------------------------
# Example Usage

# Configuration for the training run
# This combines model config with training hyperparameters
# Update vocab_size for training as per Python comment
train_config <- list(
  vocab_size = 50257L, # number of tokens for training
  block_size = 1024L, # max sequence length
  n_layer = 12L, # number of layers
  n_head = 12L,  # number of heads
  n_embd = 768L, # embedding dimension
  attn_dropout = 0.0, # Dropout values from Python training script
  resid_dropout = 0.0,

  # Training hyperparameters from Python script
  total_batch_size = 524288, # 2**19, ~0.5M, in number of tokens
  Batch = 64, # micro batch size
  Token = 1024, # sequence length
  max_lr = 6e-4,
  min_lr = 6e-4 * 0.1,
  warmup_steps = 715,
  max_steps = 19073, # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
  weight_decay = 0.1,
  eval_interval = 250, # Evaluate every 250 steps
  eval_iters = 20, # Number of batches for validation
  log_interval = 10, # Log training loss every 10 steps
  save_interval = 5000, # Save checkpoint every 5000 steps

  # Device and DDP setup (simplified for single GPU/CPU)
  # This part would need significant adaptation for proper R DDP
  device = if (cuda_is_available()) torch_device("cuda") else torch_device("cpu"),
  device_type = if (cuda_is_available()) "cuda" else "cpu",
  ddp = FALSE, # Set to TRUE if running with R DDP (requires setup)
  ddp_rank = 0,
  ddp_local_rank = 0,
  ddp_world_size = 1,
  master_process = TRUE # Assuming single process for now
)

# Calculate gradient accumulation steps
if (train_config$total_batch_size %% (train_config$Batch * train_config$Token * train_config$ddp_world_size) != 0) {
  stop("make sure total_batch_size is divisible by Batch * Token * ddp_world_size")
}
grad_accum_steps <- train_config$total_batch_size %/% (train_config$Batch * train_config$Token * train_config$ddp_world_size)
if (train_config$master_process) {
    message(sprintf("total desired batch size: %d", train_config$total_batch_size))
    message(sprintf("=> calculated gradient accumulation steps: %d", grad_accum_steps))
}


# Learning rate schedule function
get_lr <- function(it) {
    # 1) linear warmup for warmup_iters steps
    if (it < train_config$warmup_steps) {
        return(train_config$max_lr * (it + 1) / train_config$warmup_steps)
    }
    # 2) if it > max_steps, return min learning rate
    if (it > train_config$max_steps) {
        return(train_config$min_lr)
    }
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio <- (it - train_config$warmup_steps) / (train_config$max_steps - train_config$warmup_steps)
    stopifnot(decay_ratio >= 0 && decay_ratio <= 1)
    coeff <- 0.5 * (1.0 + cos(pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return(train_config$min_lr + coeff * (train_config$max_lr - train_config$min_lr))
}


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

# ---------------------------
# 1. Define weight name mapping
# ---------------------------
weight_mapping <- list(
  # Embeddings
  "wte.weight" = "transformer.wte.weight",
  "wpe.weight" = "transformer.wpe.weight",
  
  # Final layer norm
  "ln_f.weight" = "transformer.ln_f.weight",
  "ln_f.bias" = "transformer.ln_f.bias",
  
  # LM head
  "lm_head.weight" = "lm_head.weight"
)

for (i in 0:11) {
  weight_mapping[[sprintf("MH.%d.in_proj_weight", i)]] <- sprintf("transformer.h.%d.attn.c_attn.weight", i)
  weight_mapping[[sprintf("MH.%d.in_proj_bias", i)]] <- sprintf("transformer.h.%d.attn.c_attn.bias", i)
  weight_mapping[[sprintf("MH.%d.out_proj.weight", i)]] <- sprintf("transformer.h.%d.attn.c_proj.weight", i)
  weight_mapping[[sprintf("MH.%d.out_proj.bias", i)]] <- sprintf("transformer.h.%d.attn.c_proj.bias", i)
  
  weight_mapping[[sprintf("scale1.%d.weight", i)]] <- sprintf("transformer.h.%d.ln_1.weight", i)
  weight_mapping[[sprintf("scale1.%d.bias", i)]]   <- sprintf("transformer.h.%d.ln_1.bias", i)
  weight_mapping[[sprintf("scale2.%d.weight", i)]] <- sprintf("transformer.h.%d.ln_2.weight", i)
  weight_mapping[[sprintf("scale2.%d.bias", i)]]   <- sprintf("transformer.h.%d.ln_2.bias", i)

  weight_mapping[[sprintf("FFN.%d.0.weight", i)]] <- sprintf("transformer.h.%d.mlp.c_fc.weight", i)
  weight_mapping[[sprintf("FFN.%d.0.bias", i)]]   <- sprintf("transformer.h.%d.mlp.c_fc.bias", i)
  weight_mapping[[sprintf("FFN.%d.2.weight", i)]] <- sprintf("transformer.h.%d.mlp.c_proj.weight", i)
  weight_mapping[[sprintf("FFN.%d.2.bias", i)]]   <- sprintf("transformer.h.%d.mlp.c_proj.bias", i)
}

# ---------------------------
# 2. Process attention weights
# ---------------------------
process_attention_weights <- function(original_weights) {
  new_state_dict <- list()
  
  for (old_name in names(original_weights)) {
    if (grepl("MH\\.\\d+\\.in_proj_weight", old_name)) {
      layer_num <- as.numeric(gsub("MH\\.(\\d+)\\.in_proj_weight", "\\1", old_name))
      combined_weights <- original_weights[[old_name]]
      n_embd <- combined_weights$size(2)
      
      q_weights <- combined_weights[1:n_embd, ]
      k_weights <- combined_weights[(n_embd + 1):(2 * n_embd), ]
      v_weights <- combined_weights[(2 * n_embd + 1):(3 * n_embd), ]
      
      new_name <- sprintf("transformer.h.%d.attn.c_attn.weight", layer_num)
      new_state_dict[[new_name]] <- torch_cat(list(q_weights, k_weights, v_weights), dim = 1)
      
    } else if (grepl("MH\\.\\d+\\.in_proj_bias", old_name)) {
      layer_num <- as.numeric(gsub("MH\\.(\\d+)\\.in_proj_bias", "\\1", old_name))
      combined_bias <- original_weights[[old_name]]
      n_embd <- combined_bias$size(1) %/% 3
      
      q_bias <- combined_bias[1:n_embd]
      k_bias <- combined_bias[(n_embd + 1):(2 * n_embd)]
      v_bias <- combined_bias[(2 * n_embd + 1):(3 * n_embd)]
      
      new_name <- sprintf("transformer.h.%d.attn.c_attn.bias", layer_num)
      new_state_dict[[new_name]] <- torch_cat(list(q_bias, k_bias, v_bias), dim = 1)
      
    } else if (old_name %in% names(weight_mapping)) {
      new_state_dict[[weight_mapping[[old_name]]]] <- original_weights[[old_name]]
    }
  }
  
  return(new_state_dict)
}

# ---------------------------
# 3. Remap and Load Weights
# ---------------------------
remapped_state_dict <- process_attention_weights(original_weights)

# Instantiate model
test_model <- GPT(config = loaded_config)$to(device = loaded_config$device)

# Sanity check
expected_params <- names(test_model$state_dict())
missing <- setdiff(expected_params, names(remapped_state_dict))
if (length(missing) > 0) {
  warning(sprintf("Missing parameters: %s", paste(missing, collapse = ", ")))
} else {
  message("All parameters accounted for in remapped state dict.")
}

# Load remapped weights
test_model$load_state_dict(remapped_state_dict)

# Verify blocks are accessible
if (length(test_model$transformer[["h"]]) > 0) {
  message("First block accessible: TRUE")
} else {
  warning("No transformer blocks found!")
}

# ---------------------------
# 4. Link lm_head and wte AFTER loading
# ---------------------------
test_model$transformer[["wte"]]$weight <- test_model$lm_head$weight

message("Model weights loaded and tied successfully.")

# Set the model to evaluation mode
test_model$eval()

# --- Simple Inference Test ---
message("Performing a simple inference test...")

# Example prompt
prompt_text <- "Hello, I'm a language model,"
max_length <- 32 # Maximum length of generated text
num_return_sequences <- 4 # Number of sequences to generate

# Encode the prompt text
tokens <- enc$encode(prompt_text)
print(paste("Original tokens:", paste(tokens, collapse = ", ")))

# Convert to 1-based indexing for R torch
tokens_1based <- torch_tensor(matrix(unlist(tokens) + 1L, nrow = 1), dtype = torch_long())

# Add a batch dimension and move to device
xgen <- tokens_1based$to(device = loaded_config$device)

# Generate text
with_no_grad({
  while (xgen$size(2) < max_length) {
    # Get input window (handle block size limit)
    current_length <- xgen$size(2)
    if (current_length > loaded_config$block_size) {
      input_window <- xgen[, (current_length - loaded_config$block_size + 1):current_length]
    } else {
      input_window <- xgen
    }
    
    # Get logits
    logits <- test_model(input_window)$logits
    
    # Get last token logits
    last_logits <- logits[, logits$size(2), ] # shape: [batch_size, vocab_size]
    
    # Sample next token
    probs <- nnf_softmax(last_logits, dim = -1)
    next_token <- torch_multinomial(probs, num_samples = 1)
    
    # Append to sequence (next_token is already 1-based from model output)
    xgen <- torch_cat(list(xgen, next_token), dim = 2)
  }
})

# Decode and print - convert back to 0-based for the tokenizer
generated_tokens_1based <- as.integer(xgen$to(device = "cpu")[1, ])
generated_tokens_0based <- generated_tokens_1based - 1L

message(paste("Generated tokens:", paste(generated_tokens_0based, collapse = ", ")))

# Decode with tiktoken (expects 0-based tokens)
decoded_text <- enc$decode(as.list(generated_tokens_0based))
message(sprintf("Generated text: %s", decoded_text))