tryCatch({
  message("\n=== STEP 0: Installing required R packages ===")
  if (!requireNamespace("renv", quietly=TRUE)){
    install.packages("renv")
  }
  renv::restore()
 
  message("\n=== STEP 1: Setting up Python environment for reticulate ===")
  library(reticulate)
 
  python_env <- "gpt2-env"
  python_bin <- Sys.which("python3")
 
  if (python_bin == "") {
    stop("Python 3 not found. Ensure it's installed and in PATH.")
  }
 
  # Create the virtualenv if needed
  if (!virtualenv_exists(python_env)) {
    message("Creating virtual environment: ", python_env)
    virtualenv_create(python_env, python = python_bin)
  }
 
  use_virtualenv(python_env, required = TRUE)
 
  message("Using Python at: ", py_config()$python)
 
  message("\n=== STEP 2: Installing Python packages ===")
  py_install(c("numpy", "tiktoken"), envname = python_env, pip = TRUE)
  np <- import("numpy")
  tiktoken <- import("tiktoken")
  message("NumPy version: ", np$`__version__`)
  message("Tiktoken loaded: ", !is.null(tiktoken))
 
  message("\n=== STEP 3: Installing R torch package ===")
  # Sys.setenv(CUDA="12.4")
  # torch::install_torch(cuda_version = "12.4")
  torch::install_torch(type = "cpu")
  library(torch)

  message("\n=== STEP 4: Load additional R libraries ===")
  suppressMessages({
    library(R6)
    library(stringr)
    library(curl)
  })
})

