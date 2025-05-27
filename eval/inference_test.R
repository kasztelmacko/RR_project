generate_from_prompt <- function(model, prompt_text, max_length, config) {
  message("Performing a simple inference test...")

  tokens <- enc$encode(prompt_text)
  print(paste("Original tokens:", paste(tokens, collapse = ", ")))
  tokens_1based <- torch_tensor(matrix(unlist(tokens) + 1L, nrow = 1), dtype = torch_long())$to(device = config$device)

  xgen <- tokens_1based

  with_no_grad({
    while (xgen$size(2) < max_length) {
      current_length <- xgen$size(2)
      input_window <- if (current_length > config$block_size) {
        xgen[, (current_length - config$block_size + 1):current_length]
      } else {
        xgen
      }

      logits <- model(input_window)$logits
      last_logits <- logits[, logits$size(2), ]
      probs <- nnf_softmax(last_logits, dim = -1)
      next_token <- torch_multinomial(probs, num_samples = 1)

      xgen <- torch_cat(list(xgen, next_token), dim = 2)
    }
  })

  generated_tokens <- as.integer(xgen$to(device = "cpu")[1, ]) - 1L
  decoded_text <- enc$decode(as.list(generated_tokens))
  message(sprintf("Generated text: %s", decoded_text))
  return(decoded_text)
}
