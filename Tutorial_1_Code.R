# Install and load MASS package
if (!requireNamespace("MASS", quietly = TRUE)) {
  install.packages("MASS")
}
library(MASS)

# Set parameters
num_simulations <- 1000  # Number of simulations
test_sample_size <- 10000

sample_sizes <- c(1000, 2000, 5000, 10000)

num_regressors <- c(30,30,30,30) #First we keep the number of regressors fixed

#num_regressors <- c(30,45,70,100) #Now we let them grow at roughly the square root of the sample size 

#num_regressors <- c(100,200,500,1000) #Now we use a fixed proportion of the sample size 


# Initialize vectors to store results
mse_values <- numeric(length(num_regressors))

# Monte Carlo simulation
for (k in seq_along(num_regressors)) {
  p <- num_regressors[k] #This sets the number of regressors to the k-th entry in num_regressors
  n <- sample_sizes[k] #This sets the number of observations to the k-th entry in sample_sizes
  mse_sim <- numeric(num_simulations)
  cat(p, "\n") #So we can check our progress
  
  for (sim in 1:num_simulations) {
    
    #The sample variance covariance matrix of X  and U is a sufficient
    #statistic. We draw this directly
    S_train <- matrix(rWishart(1, n, diag(p + 1)/n), nrow = p + 1, ncol = p + 1)
    S_test <- matrix(rWishart(1, test_sample_size, diag(p + 1))/test_sample_size, nrow = p + 1, ncol = p + 1)
    
    XX_train <- S_train[1:p, 1:p]     #This is X'X/n on the training sample
    XU_train <- S_train[1:p, (p + 1):(p + 1)]      #This is X'U/n on the training sample
    XX_test <- S_test[1:p, 1:p]      #This is X'X/n on the test sample
    XU_test <- S_test[1:p, (p + 1):(p + 1)]       #This is X'U/n on the test sample
    
    beta_true <- rnorm(p)  #We draw the true coefficients randomly  
    XY_train <- XX_train %*% beta_true + XU_train   #Form X'Y/n on the training sample
    XY_test <- XX_test %*% beta_true + XU_test    #Form X'Y/n on the test sample
    
    beta_hat <- solve(XX_train, XY_train)     #calculate the OLS estimates
    
    #Finally we calculate the MSE of the estimates on the test sample relative to the true model
    mse_sim[sim] <- -2 * t(XY_test) %*% (beta_hat - beta_true) +
      t(beta_hat) %*% XX_test %*% beta_hat - t(beta_true) %*% XX_test %*% beta_true 
    
  }
  #We take the mean over simulation draws
  mse_values[k] <- mean(mse_sim)
}

# Plot the results
plot(sample_sizes, mse_values, type = "b", col = "blue", lwd = 2,
     xlab = "Sample Size", ylab = "Mean-Squared Error",
     main = "Monte Carlo Simulation of MSE")
