library(jsonlite)
library(httr)
library(progress)
library(stringr)
library(purrr)
library(reticulate)

virtualenv_create("r-reticulate")
#Python packages - if there is an error please add versions
virtualenv_install("r-reticulate", packages = c("torch", "transformers", "tqdm", "requests"))
use_virtualenv("r-reticulate", required = TRUE)
#Python modules
torch <- import("torch")
transformers <- import("transformers")
tqdm <- import("tqdm")
requests <- import("requests")
json <- import("json")
os <- import("os")

#hellaswag
data_cache_dir <- file.path(getwd(), "hellaswag")
dir.create(data_cache_dir, showWarnings = FALSE)

hellaswags <- list(
  train = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
  val = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
  test = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl"
)

#downloading a file
download_file <- function(url, fname) {
  response <- requests$get(url, stream = TRUE)
  total <- as.integer(response$headers$get("content-length", 0))
  file <- file(fname, "wb")
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  downloaded <- 0
  for (chunk in response$iter_content(chunk_size = 1024)) {
    writeBin(chunk, file)
    downloaded <- downloaded + length(chunk)
    setTxtProgressBar(pb, downloaded)
  }
  close(pb)
  close(file)
}

#downloading the dataset split
download_split <- function(split) {
  data_url <- hellaswags[[split]]
  data_filename <- file.path(data_cache_dir, paste0("hellaswag_", split, ".jsonl"))
  if (!file.exists(data_filename)) {
    cat(sprintf("Downloading %s to %s...\n", data_url, data_filename))
    download_file(data_url, data_filename)
  }
}
#rendering

tokenizer <- transformers$GPT2TokenizerFast$from_pretrained("gpt2")

render_example <- function(example) {
  ctx <- example$ctx
  label <- as.integer(example$label)
  endings <- example$endings
  
  ctx_tokens <- tokenizer$encode(ctx, add_special_tokens = FALSE)
  tok_rows <- list()
  mask_rows <- list()
  for (end in endings) {
    end_tokens <- tokenizer$encode(paste0(" ", end), add_special_tokens = FALSE)
    tok_row <- c(ctx_tokens, end_tokens)
    mask_row <- c(rep(0L, length(ctx_tokens)), rep(1L, length(end_tokens)))
    tok_rows <- append(tok_rows, list(tok_row))
    mask_rows <- append(mask_rows, list(mask_row))
  }
  
  max_len <- max(sapply(tok_rows, length))
  tokens <- torch$zeros(c(4L, as.integer(max_len)), dtype = torch$long)
  mask <- torch$zeros(c(4L, as.integer(max_len)), dtype = torch$long)
  for (i in 1:4) {
    tokens[i - 1, 0:(length(tok_rows[[i]]) - 1)] <- torch$tensor(tok_rows[[i]], dtype = torch$long)
    mask[i - 1, 0:(length(mask_rows[[i]]) - 1)] <- torch$tensor(mask_rows[[i]], dtype = torch$long)
  }
  
  list(tokens = tokens, mask = mask, label = label)
}

#evaluation

evaluate <- function(model_type = "gpt2", device = "cuda") {
  torch$set_float32_matmul_precision("high")
  model <- transformers$GPT2LMHeadModel$from_pretrained(model_type)
  model$to(device)
  model$eval()
  
  num_correct_norm <- 0
  num_correct <- 0
  num_total <- 0
  
  val_file <- file.path(data_cache_dir, "hellaswag_val.jsonl")
  con <- file(val_file, "r")
  on.exit(close(con))
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    example <- json$loads(line)
    rendered <- render_example(example)
    tokens <- rendered$tokens$to(device)
    mask <- rendered$mask$to(device)
    label <- rendered$label
    
    with_no_grad({
      outputs <- model(tokens)
      logits <- outputs$logits
    })
    
    shift_logits <- logits[ , 1:(logits$size(2) - 1), ]$contiguous()
    shift_tokens <- tokens[ , 2:tokens$size(2)]$contiguous()
    flat_shift_logits <- shift_logits$view(c(-1L, shift_logits$size(-1)))
    flat_shift_tokens <- shift_tokens$view(-1L)
    shift_losses <- nnf_cross_entropy(flat_shift_logits, flat_shift_tokens, reduction = "none")
    shift_losses <- shift_losses$view(c(tokens$size(0), -1L))
    shift_mask <- mask[ , 2:mask$size(2)]$contiguous()
    masked_shift_losses <- shift_losses * shift_mask
    sum_loss <- masked_shift_losses$sum(dim = 1)
    avg_loss <- sum_loss / shift_mask$sum(dim = 1)
    pred <- as.integer(torch$argmin(sum_loss)$item())
    pred_norm <- as.integer(torch$argmin(avg_loss)$item())
    
    num_total <- num_total + 1
    num_correct <- num_correct + as.integer(pred == label)
    num_correct_norm <- num_correct_norm + as.integer(pred_norm == label)
    cat(sprintf("%d acc_norm: %d/%d=%.4f\n", num_total, num_correct_norm, num_total, num_correct_norm / num_total))
    
    if (num_total < 10) {
      cat("---\n")
      cat(sprintf("Context:\n %s\n", example$ctx))
      cat("Endings:\n")
      for (i in 1:4) {
        cat(sprintf("%d (loss: %.4f) %s\n", i - 1, avg_loss[i - 1]$item(), example$endings[[i]]))
      }
      cat(sprintf("Predicted: %d, Actual: %d\n", pred_norm, label))
    }
  }
}

download_split("val")
evaluate(model_type = "gpt2", device = "cuda")