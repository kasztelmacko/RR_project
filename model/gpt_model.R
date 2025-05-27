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
    pos <- torch_arange(1, Token, dtype = torch_long(), device = idx$device)

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
