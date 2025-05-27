tryCatch({
  message("\n=== STEP 0: Installing required R packages ===")
  
  # Ensure VS Code R dependencies are installed
  required_packages <- c("jsonlite", "rlang", "pacman", "reticulate", "R6", "remotes", "curl", "stringr", "torch")
  
  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message("Installing ", pkg, "...")
      install.packages(pkg, repos = "https://cloud.r-project.org")
    } else {
      message(pkg, " is already installed")
    }
  }
 
  # Load pacman for easier package management
  if (!requireNamespace("pacman", quietly = TRUE)) {
    install.packages("pacman", repos = "https://cloud.r-project.org")
  }
 
  pacman::p_load(reticulate, R6)
 
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
 
  message("\n=== STEP 3: Installing R torch package with CUDA 12.4 support ===")
  Sys.setenv(CUDA="12.4")
  torch::install_torch(type = "cpu")
  library(torch)
})
  

library(jsonlite)
library(curl)
library(stringr)