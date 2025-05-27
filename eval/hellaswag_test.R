render_example <- function(example, enc) {
  cat("DEBUG: Starting render_example\n")
  
  ctx <- example$ctx
  label <- as.integer(example$label)
  endings <- example$endings
  
  cat(sprintf("DEBUG: Context length: %d, Label: %d, Number of endings: %d\n", 
              nchar(ctx), label, length(endings)))
 
  # Encode context tokens (0-based from tiktoken)
  ctx_tokens <- enc$encode(ctx)
  cat(sprintf("DEBUG: Context tokens: %s\n", paste(ctx_tokens, collapse=", ")))
  cat(sprintf("DEBUG: Context tokens length: %d\n", length(ctx_tokens)))

  tok_rows <- list()
  mask_rows <- list()
  for (i in seq_along(endings)) {
    end <- endings[[i]]
    cat(sprintf("DEBUG: Processing ending %d: '%s'\n", i, substr(end, 1, 50)))
    
    # Encode ending tokens (0-based from tiktoken)
    end_tokens <- enc$encode(paste0(" ", end))
    cat(sprintf("DEBUG: Ending %d tokens: %s\n", i, paste(end_tokens, collapse=", ")))
    cat(sprintf("DEBUG: Ending %d tokens length: %d\n", i, length(end_tokens)))
    
    tok_row <- c(ctx_tokens, end_tokens)
    mask_row <- c(rep(0L, length(ctx_tokens)), rep(1L, length(end_tokens)))
    
    cat(sprintf("DEBUG: Ending %d total tokens: %d, mask sum: %d\n", 
                i, length(tok_row), sum(mask_row)))
    
    tok_rows[[i]] <- tok_row
    mask_rows[[i]] <- mask_row
  }
 
  max_len <- max(sapply(tok_rows, length))
  cat(sprintf("DEBUG: Max sequence length: %d\n", max_len))
  
  cat("DEBUG: About to create torch tensors\n")
  
  tryCatch({
    # Convert to 1-based indices for PyTorch
    tokens_matrix <- t(sapply(tok_rows, function(row) {
      c(row + 1L, rep(0L, max_len - length(row)))  # 0 is padding
    }))
    mask_matrix <- t(sapply(mask_rows, function(row) {
      c(row, rep(0L, max_len - length(row)))
    }))

    # Additional debug output
    cat("DEBUG: First row tokens (1-based):", paste(tokens_matrix[1,], collapse=", "), "\n")
    cat("DEBUG: First row mask:", paste(mask_matrix[1,], collapse=", "), "\n")

    tokens <- torch_tensor(tokens_matrix, dtype = torch_long())
    mask <- torch_tensor(mask_matrix, dtype = torch_long())
  }, error = function(e) {
    cat(sprintf("DEBUG: Error building tensors: %s\n", e$message))
    stop(e)
  })
  
  cat("DEBUG: render_example completed successfully\n")
  return(list(tokens = tokens, mask = mask, label = label))
}

hellaswag_test <- function(model, enc, config, max_examples = 10) {
  cat("DEBUG: Starting hellaswag_test\n")
  cat(sprintf("DEBUG: Config device: %s\n", config$device))
 
  num_correct_norm <- 0
  num_total <- 0
 
  val_file <- file.path(data_cache_dir, "hellaswag_val.jsonl")
  cat(sprintf("DEBUG: Reading from file: %s\n", val_file))
  
  con <- file(val_file, "r")
  on.exit(close(con))
 
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0 && num_total < max_examples) {
    cat(sprintf("\nDEBUG: Processing example %d\n", num_total + 1))
    
    tryCatch({
      example <- fromJSON(line)
      cat("DEBUG: Successfully parsed JSON\n")
    }, error = function(e) {
      cat(sprintf("DEBUG: Error parsing JSON: %s\n", e$message))
      stop(e)
    })
    
    tryCatch({
      rendered <- render_example(example, enc)
      cat("DEBUG: Successfully rendered example\n")
    }, error = function(e) {
      cat(sprintf("DEBUG: Error in render_example: %s\n", e$message))
      stop(e)
    })
   
    tryCatch({
      tokens <- rendered$tokens$to(device = config$device)
      mask <- rendered$mask$to(device = config$device)
      label <- rendered$label
      cat("DEBUG: Successfully moved tensors to device\n")
    }, error = function(e) {
      cat(sprintf("DEBUG: Error moving tensors to device: %s\n", e$message))
      stop(e)
    })
   
    tryCatch({
      with_no_grad({
        logits <- model(tokens)$logits
        
        # Calculate shift for next-token prediction
        shift_logits <- logits[, 1:(logits$size(2) - 1), ]$contiguous()
        shift_tokens <- tokens[, 2:tokens$size(2)]$contiguous()
        shift_mask <- mask[, 2:mask$size(2)]$contiguous()
        
        # Create combined mask that excludes padding and focuses on completion tokens
        combined_mask <- (shift_tokens != 0) * shift_mask
        
        # Only calculate loss for non-padding, masked positions
        flat_shift_logits <- shift_logits[combined_mask == 1]
        flat_shift_tokens <- shift_tokens[combined_mask == 1]
        
        # Reshape for loss calculation
        flat_shift_logits <- flat_shift_logits$view(c(tokens$size(0), -1, shift_logits$size(-1)))
        flat_shift_tokens <- flat_shift_tokens$view(c(tokens$size(0), -1))
        
        # Calculate cross entropy loss
        shift_losses <- nnf_cross_entropy(
          flat_shift_logits$transpose(1, 2),
          flat_shift_tokens,
          reduction = "none"
        )
        
        # Calculate average loss for each completion
        avg_loss <- shift_losses$mean(dim = 2)
        
        # Prediction is the completion with lowest average loss
        pred_norm <- as.integer(torch_argmin(avg_loss)$item())
      })
      num_correct_norm <- num_correct_norm + as.integer(pred_norm == label)
      num_total <- num_total + 1
    }, error = function(e) {
      cat(sprintf("DEBUG: Error during inference block: %s\n", e$message))
      stop(e)
    })
   
    cat(sprintf("%d acc_norm: %d/%d = %.4f\n", num_total, num_correct_norm, num_total, num_correct_norm / num_total))
   
    cat("---\n")
    cat(sprintf("Context:\n%s\n", example$ctx))
    for (i in 1:4) {
      cat(sprintf("%d (loss: %.4f) %s\n", i - 1, avg_loss[i]$item(), example$endings[[i]]))
    }
    cat(sprintf("Predicted: %d, Actual: %d\n\n", pred_norm, label))
  }
 
  return(list(accuracy = num_correct_norm / num_total,
              correct = num_correct_norm,
              total = num_total))
}