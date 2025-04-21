install.packages("Rcpp")
library(Rcpp)
install.packages("devtools")
install.packages("usethis")
library(usethis)
create_package("MonteCarloOptionPricing")

Sys.setenv(PKG_CXXFLAGS = "-I/usr/local/include/Rcpp")
devtools::clean_dll()    # Clean compiled files (if any)
devtools::load_all()     # Rebuild the package
# Test the function from your package
library(MonteCarloOptionPricing)
library(MonteCarloOptionPricing)

# Example usage
monteCarloOptionPrice(S0 = 105, K = 110, sigma = 0.21, r = 0.05, t = 0.75, n = 10000, b = 115)

result <- monteCarloOptionPrice(S0 = 100, K = 95, sigma = 0.2, r = 0.05, t = 1, n = 10000, b = 120)
print(result)

S0 <- 105  
K <- 110
sigma <- 0.21
r <- 0.05
t <- 0.75
n <- 10000
b <- 115

cppFunction('
double monteCarloOptionPrice(double S0, double K, double sigma, double r, double t, int n, double barrier) {
  double payoff = 0.0;
  double ST, Z;
  srand(time(0)); // Seed random number generator for reproducibility
  
  // Monte Carlo simulation loop
  for (int i = 0; i < n; ++i) {
    // Generate random value for normal distribution (standard normal)
    Z = (double) rand() / RAND_MAX;  // Uniform random number
    Z = sqrt(-2.0 * log(Z)) * cos(2.0 * M_PI * rand() / RAND_MAX); // Box-Muller transform
    
    // Simulate the options final price ST using geometric Brownian motion
            ST = S0 * exp((r - 0.5 * sigma * sigma) * t + sigma * sqrt(t) * Z);
            
            // Check if the barrier condition is hit
            if (ST < barrier) {
              payoff += std::max(ST - K, 0.0);  // Option payoff, European call style
            } else {
              payoff += 0.0;  // Barrier condition has been hit, no payoff
            }
            }

// Return the average payoff discounted by the risk-free rate
return exp(-r * t) * payoff / n;
}
')


option_price <- monteCarloOptionPrice(S0, K, sigma, r, t, n, barrier)
print(paste("Option Price: ", option_price))

barrier_levels <- seq(100, 120, by = 1)
prices <- sapply(barrier_levels, function(b) monteCarloOptionPrice(S0, K, sigma, r, t, n, b))

plot(barrier_levels, prices, type = "l", col = "cyan4", 
     xlab = "Barrier Level", ylab = "Option Price", 
     main = "Monte Carlo option pricing with barrier levels")



Rcpp::Rcpp.package.skeleton(name = "DownInOption", path = ".")

# Step 2: Write C++ Code for Monte Carlo Simulation
# This code will be added to the package's src/ folder as monte_carlo.cpp

library(Rcpp)
system("echo \"#include <Rcpp.h>\nusing namespace Rcpp;\n\n// [[Rcpp::export]]\ndouble monte_carlo_option_price(double S0, double K, double sigma, double r, double T, double barrier, int n_paths) {\n    NumericVector payoffs(n_paths);\n    double dt = T / 252.0;\n    for (int i = 0; i < n_paths; i++) {\n        double S = S0;\n        bool breached = false;\n        for (int j = 0; j < 252; j++) {\n            S *= exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * R::rnorm(0, 1));\n            if (S <= barrier) {\n                breached = true;\n            }\n        }\n        payoffs[i] = breached ? std::max(S - K, 0.0) : 0.0;\n    }\n    return mean(payoffs) * exp(-r * T);\n}\n\" > DownInOption/src/monte_carlo.cpp")

# Compile the package
system("R CMD INSTALL DownInOption")

# Step 3: R Markdown Report Template
cat("---\ntitle: \"Down-and-In Option Pricing\"\nauthor: \"Joanna Jędrzejewska\"\noutput: html_document\n---\n\n```{r setup, include=FALSE}\nknitr::opts_chunk$set(echo = TRUE)\n```\n\n## Objective\nThis report computes the theoretical price of a down-and-in call option using Monte Carlo simulation.\n\n## Assumptions\n- Geometric Brownian motion for asset price dynamics.\n- Parameters: \n  - Initial Price: \(S_0 = 105\)\n  - Strike Price: \(K = 110\)\n  - Volatility: \(\sigma = 0.21\)\n  - Risk-free Rate: \(r = 0.05\)\n  - Time to Maturity: \(T = 0.75\)\n  - Barrier: \(b = 100\)\n- Monte Carlo Iterations: 10,000 paths.\n\n## Simulation Results\n\n```{r}
library(DownInOption)
price <- monte_carlo_option_price(S0 = 105, K = 110, sigma = 0.21, r = 0.05, T = 0.75, barrier = 100, n_paths = 10000)
price\n```\n\n## Volatility and Maturity Analysis\n```{r}
volatilities <- seq(0.1, 0.5, by = 0.05)
maturities <- seq(0.25, 1, by = 0.25)
results <- expand.grid(volatility = volatilities, maturity = maturities)
results$price <- mapply(monte_carlo_option_price, 105, 110, results$volatility, 0.05, results$maturity, 100, 10000)

library(ggplot2)
ggplot(results, aes(x = volatility, y = price, color = factor(maturity))) +
    geom_line() +
    labs(title = \"Option Price vs. Volatility and Maturity\", x = \"Volatility\", y = \"Option Price\") +
    theme_minimal()
```\n", file = "DownInOption_Report.Rmd")
