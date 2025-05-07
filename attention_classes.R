library(torch)
library(R6)

# Configuration for the GPT mode
GPTConfig <- list(
  block_size = 1024L, # max sequence length
  vocab_size = 50257L, # for model training is 50304L
  n_layer = 12L, # number of layers
  n_head = 12L,  # number of heads
  n_embd = 768L  # embedding dimension
)


library(torch)

scaled_dot_product_attention <- function(query, key, value,
                                         is_causal = FALSE, dropout_p = 0.0) {
  # Partial implementation of the python version
  # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

  L <- query$size(-2)
  S <- key$size(-2)

  scale_factor <- 1 / sqrt(query$size(-1))

  attn_bias <- torch_zeros(c(L, S), dtype = query$dtype, device = query$device)

  if (is_causal) {
    temp_mask <- torch_ones(c(L, S), dtype = torch_bool(), device = query$device)$tril(diagonal = 0)
    attn_bias <- attn_bias$masked_fill(temp_mask$logical_not(), -Inf)
    attn_bias <- attn_bias$to(dtype = query$dtype)
  }

  attn_weight <- torch_matmul(query, key$transpose(-2, -1)) * scale_factor
  attn_weight <- attn_weight + attn_bias
  attn_weight <- nnf_softmax(attn_weight, dim = -1)

  if (dropout_p > 0.0) {
    attn_weight <- nnf_dropout(attn_weight, p = dropout_p, training = TRUE)
  }

  output <- torch_matmul(attn_weight, value)

  return(output)
}


CausalSelfAttention <- nn_module(
  "CausalSelfAttention",
  initialize = function(config) {
    super$initialize() 
    stopifnot(config$n_embd %% config$n_head == 0)
    
    # key, query, value projections for all heads, but in a batch
    self$c_attn <- nn_linear(config$n_embd, 3 * config$n_embd)
    # output projection
    self$c_proj <- nn_linear(config$n_embd, config$n_embd)
    self$c_proj$NANOGPT_SCALE_INIT <- 1 
    
    # regularization
    self$n_head <- config$n_head
    self$n_embd <- config$n_embd
    
  },
  forward = function(x) {
    # B = batch size, T = sequence length, C = embedding dimensionality (n_embd)
    dims <- x$size()
    B <- dims[[1]]
    T <- dims[[2]]
    C <- dims[[3]]
    
    # Calculate query, key, values for all heads in batch
    qkv <- self$c_attn(x)
    qkv_splitted <- qkv$split(self$n_embd, dim = 3) # Split along the embedding dimension
    q <- qkv_splitted[[1]]
    k <- qkv_splitted[[2]]
    v <- qkv_splitted[[3]]
    
    # Reshape and transpose
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    
    head_size <- C %/% self$n_head
    k <- k$view(c(B, T, self$n_head, head_size))$transpose(2,3)
    q <- q$view(c(B, T, self$n_head, head_size))$transpose(2,3)
    v <- v$view(c(B, T, self$n_head, head_size))$transpose(2,3)
    
    y <- scaled_dot_product_attention(q, k, v, is_causal = TRUE)
    y <- y$transpose(2, 3)$contiguous()$view(c(B, T, C))
    y <- self$c_proj(y)
    
    return(y)
  }
)


MLP <- nn_module(
  "MLP",
  initialize = function(config) {
    super$initialize() 
    self$c_fc   <- nn_linear(config$n_embd, 4 * config$n_embd)
    self$gelu   <- nn_gelu(approximate='tanh')
    self$c_proj <- nn_linear(4 * config$n_embd, config$n_embd)
    self$c_proj$NANOGPT_SCALE_INIT <- 1
  },
  forward = function(x) {
    x <- self$c_fc(x)
    x <- self$gelu(x)
    x <- self$c_proj(x)
    return(x)
  }
)


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
    x <- x + self$attn(self$ln_1(x))
    x <- x + self$mlp(self$ln_2(x))
    return(x)
  }
)

