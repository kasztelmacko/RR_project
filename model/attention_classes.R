# -----------------------------------------------------------------------------
# Scaled Dot Product Attention (from previous turn, included for completeness)
scaled_dot_product_attention <- function(query, key, value,
                                         is_causal = FALSE, dropout_p = 0.0) {
  # Partial implementation of the python version
  # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

  L <- query$size(-2)
  S <- key$size(-2)

  scale_factor <- 1 / sqrt(query$size(-1))

  # attn_bias needs to be on the same device as query
  attn_bias <- torch_zeros(c(L, S), dtype = query$dtype, device = query$device)

  if (is_causal) {
    # temp_mask needs to be on the same device as query
    temp_mask <- torch_ones(c(L, S), dtype = torch_bool(), device = query$device)$tril(diagonal = 0)
    attn_bias <- attn_bias$masked_fill(temp_mask$logical_not(), -Inf)
    attn_bias <- attn_bias$to(dtype = query$dtype)
  }

  attn_weight <- torch_matmul(query, key$transpose(-2, -1)) * scale_factor
  attn_weight <- attn_weight + attn_bias
  attn_weight <- nnf_softmax(attn_weight, dim = -1)

  # Note: Dropout should only be applied during training
  if (dropout_p > 0.0 && query$training) { # Check if the module is in training mode
    attn_weight <- nnf_dropout(attn_weight, p = dropout_p)
  }

  output <- torch_matmul(attn_weight, value)

  return(output)
}

# -----------------------------------------------------------------------------
# Causal Self Attention (Completing the R6 class)
CausalSelfAttention <- nn_module(
  "CausalSelfAttention",
  initialize = function(config) {
    super$initialize()
    stopifnot(config$n_embd %% config$n_head == 0)

    # key, query, value projections for all heads, but in a batch
    self$c_attn <- nn_linear(config$n_embd, 3 * config$n_embd)
    # output projection
    self$c_proj <- nn_linear(config$n_embd, config$n_embd)
    # The NANOGPT_SCALE_INIT attribute is a custom marker used in the Python code
    # for specific initialization logic. We can attach it as an attribute in R.
    attr(self$c_proj, "NANOGPT_SCALE_INIT") <- 1

    # regularization (dropout probability for attention weights)
    # Assuming dropout is part of the config or a separate parameter
    self$attn_dropout <- config$attn_dropout # Add attn_dropout to your config list
    self$resid_dropout <- config$resid_dropout # Add resid_dropout to your config list

    self$n_head <- config$n_head
    self$n_embd <- config$n_embd

    # Causal mask is not needed as scaled_dot_product_attention handles it
  },
  forward = function(x) {
    # x is of shape (Batch, Token, C)
    Batch <- x$size(1) # batch size
    Token <- x$size(2) # sequence length
    C <- x$size(3) # embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    qkv <- self$c_attn(x)
    # split into q, k, v
    # Python: q, k, v = qkv.split(self.n_embd, dim=2)
    # In R, we can use `torch_split` or manual slicing/viewing
    q <- qkv[, , 1:self$n_embd]
    k <- qkv[, , (self$n_embd + 1):(2 * self$n_embd)]
    v <- qkv[, , (2 * self$n_embd + 1):(3 * self$n_embd)]

    # reshape and transpose for multi-head attention
    # Python: k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    hs <- C %/% self$n_head # head size
    k <- k$view(c(Batch, Token, self$n_head, hs))$transpose(2, 3) # R transpose(dim1, dim2)
    q <- q$view(c(Batch, Token, self$n_head, hs))$transpose(2, 3)
    v <- v$view(c(Batch, Token, self$n_head, hs))$transpose(2, 3)

    # causal self-attention; self-attend: (Batch, nh, Token, hs) x (Batch, nh, hs, Token) -> (Batch, nh, Token, Token)
    # Python: y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
    # Using our R implementation of scaled_dot_product_attention
    y <- scaled_dot_product_attention(q, k, v, is_causal = TRUE, dropout_p = self$attn_dropout)

    # re-assemble all head outputs side by side
    # Python: y = y.transpose(1, 2).contiguous().view(B, T, C)
    y <- y$transpose(2, 3)$contiguous()$view(c(Batch, Token, C))

    # output projection
    y <- self$c_proj(y)

    # Apply residual dropout after the projection
    y <- nnf_dropout(y, p = self$resid_dropout, training = self$training)

    return(y)
  }
)

# -----------------------------------------------------------------------------
# MLP (translation of Python's MLP class)
MLP <- nn_module(
  "MLP",
  initialize = function(config) {
    super$initialize()
    self$c_fc    <- nn_linear(config$n_embd, 4 * config$n_embd)
    # R's GELU is `nnf_gelu`, `approximate='tanh'` is the default
    self$gelu    <- nn_gelu() # Use nn_gelu module
    self$c_proj  <- nn_linear(4 * config$n_embd, config$n_embd)
    # Custom marker for initialization
    attr(self$c_proj, "NANOGPT_SCALE_INIT") <- 1

    # Residual dropout
    self$resid_dropout <- config$resid_dropout # Add resid_dropout to your config list
  },
  forward = function(x) {
    x <- self$c_fc(x)
    x <- self$gelu(x)
    x <- self$c_proj(x)
    # Apply residual dropout
    x <- nnf_dropout(x, p = self$resid_dropout, training = self$training)
    return(x)
  }
)

# -----------------------------------------------------------------------------
# Block (translation of Python's Block class)
Block <- nn_module(
  "Block",
  initialize = function(config) {
    super$initialize()
    self$ln_1 <- nn_layer_norm(config$n_embd)
    self$attn <- CausalSelfAttention(config)
    self$ln_2 <- nn_layer_norm(config$n_embd)
    self$mlp <- MLP(config)
  },
  forward = function(x) {
    # Python: x = x + self.attn(self.ln_1(x))
    x <- x + self$attn(self$ln_1(x))
    # Python: x = x + self.mlp(self$ln_2(x))
    x <- x + self$mlp(self$ln_2(x))
    return(x)
  }
)