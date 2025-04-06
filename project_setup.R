install.packages(c("R6", "reticulate", "torch"), repos = "https://cloud.r-project.org")

library(reticulate)

if (!virtualenv_exists("gpt2-env")) {
  virtualenv_create("gpt2-env")
}
use_virtualenv("gpt2-env", required = TRUE)

py_install(packages = c("tiktoken", "numpy", "torch"))

library(torch)
install_torch(type = "cpu") # Change to "cuda" if you have NVIDIA GPU
message("Python environment setup complete. Required packages installed.")

library(R6)
library(torch)
np <- import("numpy")
tiktoken <- import("tiktoken")
message("Project setup complete. Required libraries installed and loaded.")