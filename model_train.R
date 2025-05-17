# NOTE: Add new installs and imports to the project_setup.R file:
source('project_setup.R')

# -----------------------------------------------------------------------------
load_tokens <- function(filename) {
  npt <- np$load(filename)
  npt <- as.integer(npt)
  ptt <- torch::torch_tensor(npt, dtype = torch::torch_long())
  return(ptt)
}

DataLoaderLite <- R6Class(
  "DataLoaderLite",
  public = list(
    Batch = NULL,
    Token = NULL,
    process_rank = NULL,
    num_processes = NULL,
    split = NULL,
    shards = NULL,
    current_shard = NULL,
    tokens = NULL,
    current_position = NULL,
    initialize = function(Batch, Token, process_rank, num_processes, split, master_process = FALSE) {
      self$Batch <- Batch
      self$Token <- Token
      self$process_rank <- process_rank
      self$num_processes <- num_processes
      if (!split %in% c("train", "val")) {
        stop("split must be either 'train' or 'val'")
      }
      data_root <- "data/ptbdataset/processed"
      shards <- list.files(data_root)
      shards <- shards[grepl(split, shards)]
      shards <- sort(shards)
      shards <- file.path(data_root, shards)
      self$shards <- shards
      if (length(shards) == 0) {
        stop(paste0("no shards found for split ", split))
      }
      if (master_process) {
        message(paste0("found ", length(shards), " shards for split ", split))
      }
      self$reset()
    },
    reset = function() {
      self$current_shard <- 1
      self$tokens <- load_tokens(self$shards[self$current_shard])
      self$current_position <- self$Batch * self$Token * self$process_rank + 1
    },
    next_batch = function() {
      Batch <- self$Batch
      Token <- self$Token
      start_pos <- self$current_position
      end_pos <- start_pos + Batch * Token
      buf <- self$tokens[(start_pos - 1):(end_pos - 1)]
      x <- buf[1:(length(buf) - 1)]$view(list(Batch, Token))
      y <- buf[2:length(buf)]$view(list(Batch, Token))
      self$current_position <- self$current_position + Batch * Token * self$num_processes
      max_position <- length(self$tokens)
      if ((self$current_position + (Batch * Token * self$num_processes)) > max_position) {
        self$current_shard <- (self$current_shard %% length(self$shards)) + 1
        self$tokens <- load_tokens(self$shards[self$current_shard])
        self$current_position <- Batch * Token * self$process_rank + 1
      }
      return(list(x = x, y = y))
    }
  )
)
# -----------------------------------------------------------------------------
#model config - smaller than in Python code, lighter for experimenting purposes
#can be updated during testing
##########################################
#the python config:                      #
#n_layer	12                             #
#n_head	12                               #
#n_embd	768.                             #
#block_size	1024                         #
##########################################
config <- list(
  vocab_size = 50257,
  block_size = 128,
  n_layer = 4,
  n_head = 4,
  n_embd = 128,
  batch_size = 16,
  max_iters = 500,
  eval_interval = 50,
  learning_rate = 3e-4,
  device = if (cuda_is_available()) torch_device("cuda") else torch_device("cpu"),
  patience = 5
)

#defining GPT model
GPTModel <- nn_module(
  "GPTModel",
  initialize = function(vocab_size, n_embd, n_layer, n_head, block_size) {
    self$embedding <- nn_embedding(vocab_size, n_embd)
    self$transformer <- nn_transformer_encoder(
      encoder_layer = nn_transformer_encoder_layer(d_model = n_embd, nhead = n_head),
      num_layers = n_layer
    )
    self$linear <- nn_linear(n_embd, vocab_size)
    self$block_size <- block_size
  },
  forward = function(x) {
    x <- self$embedding(x)
    x <- x$permute(c(2, 1, 3))
    x <- self$transformer(x)
    x <- x$permute(c(2, 1, 3))
    x <- self$linear(x)
    return(x)
  }
)

#model instantiation
model <- GPTModel(
  vocab_size = config$vocab_size,
  n_embd = config$n_embd,
  n_layer = config$n_layer,
  n_head = config$n_head,
  block_size = config$block_size
)$to(device = config$device)

optimizer <- optim_adam(model$parameters, lr = config$learning_rate)
criterion <- nn_cross_entropy_loss()

train_loader <- DataLoaderLite$new(
  Batch = config$batch_size,
  Token = config$block_size,
  process_rank = 0,
  num_processes = 1,
  split = "train",
  master_process = TRUE
)

val_loader <- DataLoaderLite$new(
  Batch = config$batch_size,
  Token = config$block_size,
  process_rank = 0,
  num_processes = 1,
  split = "val",
  master_process = TRUE
)

evaluate <- function(loader) {
  model$eval()
  total_loss <- 0
  for (i in 1:10) {
    batch <- loader$next_batch()
    x <- batch$x$to(device = config$device)
    y <- batch$y$to(device = config$device)
    with_no_grad({
      logits <- model(x)
      loss <- criterion(logits$view(c(-1, config$vocab_size)), y$view(-1))
    })
    total_loss <- total_loss + loss$item()
  }
  model$train()
  return(total_loss / 10)
}

#training loop
best_val_loss <- Inf
patience_counter <- 0

for (iter in 1:config$max_iters) {
  batch <- train_loader$next_batch()
  x <- batch$x$to(device = config$device)
  y <- batch$y$to(device = config$device)
  
  optimizer$zero_grad()
  logits <- model(x)
  loss <- criterion(logits$view(c(-1, config$vocab_size)), y$view(-1))
  loss$backward()
  optimizer$step()
  
  if (iter %% config$eval_interval == 0) {
    val_loss <- evaluate(val_loader)
    cat(sprintf("Iter %d | Train Loss: %.4f | Val Loss: %.4f\n", iter, loss$item(), val_loss))
    
    if (val_loss < best_val_loss) {
      best_val_loss <- val_loss
      dir.create("checkpoints", showWarnings = FALSE)
      torch_save(model$state_dict(), "checkpoints/best_model.pt")
      patience_counter <- 0
      cat("Model improved, checkpoint saved!\n")
    } else {
      patience_counter <- patience_counter + 1
      if (patience_counter >= config$patience) {
        cat("!!!Early stopping due to no improvement!!!\n")
        break
      }
    }
  }
}
######################################################################
#steps to do/think about:                                            #
#hyperparameter tuning (already listed in the code)                  #
#shuffling the data for training                                     #
#parallelization                                                     #
#model saving with torch_save to avoid losing the best model         #
######################################################################
