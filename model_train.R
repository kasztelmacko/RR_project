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
