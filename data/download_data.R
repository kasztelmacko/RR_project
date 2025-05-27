#hellaswag
data_cache_dir <- file.path(getwd(), "data/hellaswag")
dir.create(data_cache_dir, showWarnings = FALSE)

hellaswags <- list(
  train = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
  val = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
  test = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl"
)

#downloading a file
download_file <- function(url, fname) {
  con <- curl(url, "rb")
  out <- file(fname, "wb")

  cat(sprintf("Downloading %s...\n", url))
  
  repeat {
    buf <- readBin(con, "raw", 8192)
    if (length(buf) == 0) break
    writeBin(buf, out)
  }

  close(con)
  close(out)
}


#downloading the dataset split
download_split <- function(split) {
  data_url <- hellaswags[[split]]
  data_filename <- file.path(data_cache_dir, paste0("hellaswag_", split, ".jsonl"))
  if (!file.exists(data_filename)) {
    cat(sprintf("Downloading %s to %s...\n", data_url, data_filename))
    download_file(data_url, data_filename)
  }
}