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