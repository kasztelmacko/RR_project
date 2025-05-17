# .Rprofile
if (requireNamespace("reticulate", quietly = TRUE)) {
  reticulate::use_virtualenv("gpt2-env", required = TRUE)
}

if (requireNamespace("torch", quietly = TRUE)) {
  torch::install_torch(type = "cpu")
}
