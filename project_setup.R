tryCatch({
  message("\n=== STEP 0: Installing required R packages ===")
  if (!requireNamespace("pacman", quietly = TRUE)) {
    install.packages("pacman", repos = "https://cloud.r-project.org")
  }
  pacman::p_load(reticulate, torch, R6)

  message("\n=== STEP 1: Detecting available Python versions ===")
  python_bin <- Sys.which("python3")
  python_version <- system(paste(python_bin, "--version"), intern = TRUE)

  if (python_bin == "") {
    stop("Python 3 not found. Ensure it's installed and in PATH.")
  } else {
    message("Found Python at: ", python_bin)
    message("Python version: ", python_version)
  }

  message("\n=== STEP 2: Configuring Python environment ===")
  use_python(python_bin, required = TRUE)

  if (!virtualenv_exists("gpt2-env")) {
    message("Creating virtual environment: gpt2-env")
    virtualenv_create("gpt2-env", python = python_bin)
  }

  use_virtualenv("gpt2-env", required = TRUE)

  message("\n=== STEP 3: Installing Python packages ===")
  py_install(c("tiktoken", "numpy", "torch"), envname = "gpt2-env", pip = TRUE)

  message("\n=== STEP 4: Verifying torch setup ===")
  if (!torch_is_installed()) {
    message("Installing Torch (CPU version)...")
    install_torch(type = "cpu")
  }

  message("\n=== STEP 5: Final verification ===")
  np <- import("numpy")
  tiktoken <- import("tiktoken")

  message("NumPy version: ", np$`__version__`)
  message("Tiktoken module loaded: ", if (!is.null(tiktoken)) "Yes" else "No")
  message("Python path used by reticulate: ", py_config()$python)
  message("\n✅ Setup completed successfully!")

}, error = function(e) {
  message("\n❌ ERROR: ", e$message)
  message("\n🔍 Troubleshooting steps:")
  message("1. Detected Python path: ", Sys.which("python3"))
  if ("reticulate" %in% loadedNamespaces()) {
    message("2. Reticulate Python config:\n", paste(capture.output(py_config()), collapse = "\n"))
  }
  message("3. Python binaries in /usr/bin:")
  system("ls -l /usr/bin/python* || echo 'No Python binaries found in /usr/bin'")
  message("4. Installed R packages:")
  message(paste(installed.packages()[,1], collapse = ", "))
})
