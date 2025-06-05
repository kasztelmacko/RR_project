render_example <- function(example, enc) {
  ctx <- example$ctx
  label <- as.integer(example$label)
  endings <- example$endings

  ctx_tokens <- enc$encode(ctx)

  tok_rows <- list()
  mask_rows <- list()
  for (i in seq_along(endings)) {
    end <- endings[[i]]
    end_tokens <- enc$encode(paste0(" ", end))

    tok_row <- c(ctx_tokens, end_tokens)
    mask_row <- c(rep(0L, length(ctx_tokens)), rep(1L, length(end_tokens)))

    tok_rows[[i]] <- tok_row
    mask_rows[[i]] <- mask_row
  }

  max_len <- max(sapply(tok_rows, length))

  tokens_matrix <- t(sapply(tok_rows, function(row) {
    c(row + 1L, rep(1L, max_len - length(row)))  # 1-based padding
  }))
  mask_matrix <- t(sapply(mask_rows, function(row) {
    c(row, rep(0L, max_len - length(row)))
  }))

  tokens <- torch_tensor(tokens_matrix, dtype = torch_long())
  mask <- torch_tensor(mask_matrix, dtype = torch_long())

  return(list(tokens = tokens, mask = mask, label = label))
}

hellaswag_test <- function(model, enc, config, max_examples = 10) {
  num_correct_norm <- 0
  num_total <- 0

  val_file <- file.path(data_cache_dir, "hellaswag_val.jsonl")
  con <- file(val_file, "r")
  on.exit(close(con))

  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0 && num_total < max_examples) {
    example <- jsonlite::fromJSON(line)
    rendered <- render_example(example, enc)

    tokens <- rendered$tokens$to(device = config$device)
    mask <- rendered$mask$to(device = config$device)
    label <- rendered$label

    with_no_grad({
      logits <- model(tokens)$logits

      shift_logits <- logits[, 1:(logits$size(2) - 1), ]$contiguous()
      shift_tokens <- tokens[, 2:tokens$size(2)]$contiguous()
      shift_mask <- mask[, 2:mask$size(2)]$contiguous()

      batch_size <- shift_logits$size(1)
      seq_len <- shift_logits$size(2)
      vocab_size <- shift_logits$size(3)

      flat_shift_logits <- shift_logits$view(c(batch_size * seq_len, vocab_size))
      flat_shift_tokens <- shift_tokens$view(batch_size * seq_len)

      shift_losses <- nnf_cross_entropy(
        flat_shift_logits,
        flat_shift_tokens,
        reduction = "none"
      )
      shift_losses <- shift_losses$view(c(batch_size, seq_len))
      masked_shift_losses <- shift_losses * shift_mask

      sum_loss <- masked_shift_losses$sum(dim = 2)
      avg_loss <- sum_loss / shift_mask$sum(dim = 2)

      pred_norm <- as.integer(torch_argmin(avg_loss)$item())
    })

    num_correct_norm <- num_correct_norm + as.integer(pred_norm == label)
    num_total <- num_total + 1

    cat(sprintf("%d acc_norm: %d/%d = %.4f\n", num_total, num_correct_norm, num_total, num_correct_norm / num_total))
    cat("---\n")
    cat(sprintf("Context:\n%s\n", example$ctx))
    for (i in 1:4) {
      cat(sprintf("%d (loss: %.4f) %s\n", i - 1, avg_loss[i]$item(), example$endings[[i]]))
    }
    cat(sprintf("Predicted: %d, Actual: %d\n\n", pred_norm, label))
  }

  return(list(
    accuracy = num_correct_norm / num_total,
    correct = num_correct_norm,
    total = num_total
  ))
}