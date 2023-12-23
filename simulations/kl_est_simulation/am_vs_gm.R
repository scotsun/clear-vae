mixture1 <- function(x, sd) {
  f <- (dnorm(x, mean = 1, sd) + 
          dnorm(x, mean = 3, sd) + 
          dnorm(x, mean = 10, sd) + 
          dnorm(x, mean = 12, sd)) / 4
  return(f)
}

mixture2 <- function(x, sd) {
  f <- (dnorm(x, mean = 1, sd) * 
          dnorm(x, mean = 3, sd) * 
          dnorm(x, mean = 10, sd) * 
          dnorm(x, mean = 12, sd)) ^ (1/4)
  return(f)
}

x <- seq(-5, 20, 0.05)

plot(x, mixture1(x, 1), "l", col = 'red')
lines(x, mixture2(x, 1), "l", col = 'blue')

plot(x, mixture2(x, 1), "l", col = 'blue')

plot(x, mixture1(x, 3), "l", col = 'red')
lines(x, mixture2(x, 3), "l", col = 'blue')
