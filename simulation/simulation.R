library(tidyverse)
library(patchwork)
library(pROC)
setwd("../Desktop/Duke Univeristy/Research/Autism grant/simulation/")

# N <- c(100, 1000, 300, 200)
# P <- c(1, 0.1, 0.01, 1)

## XOR gate----------------------
# illustrate xor gate
## --- --- --- --- --- --- --- --- --- --- --- --- --- 

xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)
)
ggplot(data = xor_data %>% mutate(y = as.factor(y)), aes(x=x1, y=x2, color=y)) +
  geom_point(size = 10) +
  lims(x = c(-1, 2), y = c(-1, 2)) +
  labs(color = "")


lr <- glm(factor(y) ~ x1 * x2, data = xor_data, family = binomial)

auc(roc(
  xor_data$y,
  predict(lr,
          newdata = data.frame(
            x1 = c(0, 0, 1, 1),
            x2 = c(0, 1, 0, 1)
          ),
          type = "response"),
  quiet = TRUE
))

## Data Gen ----------------------
# function to generate data
## --- --- --- --- --- --- --- --- --- --- --- --- --- 


# note: XOR gate can be explained by interaction
data_generator <- function(sd, N = c(1000, 100, 200, 300), P = c(0.01, 1, 1, 0.2)) {
  print(paste("event rate:", drop(P %*% N / sum(N))))
  X <- rbind(
    matrix(rep(c(0, 0), N[1]), nrow = N[1], ncol = 2, byrow = TRUE),
    matrix(rep(c(0, 1), N[2]), nrow = N[2], ncol = 2, byrow = TRUE),
    matrix(rep(c(1, 0), N[3]), nrow = N[3], ncol = 2, byrow = TRUE),
    matrix(rep(c(1, 1), N[4]), nrow = N[4], ncol = 2, byrow = TRUE)
  ) + 
    matrix(rnorm(2 * sum(N), mean = 0, sd = sd), nrow = sum(N), ncol = 2)
  
  cluster <- c(rep(1, N[1]), rep(2, N[2]), rep(3, N[3]), rep(4, N[4]))
  
  y <- c(
    rbinom(size = 1, prob = P[1], n = N[1]),
    rbinom(size = 1, prob = P[2], n = N[2]),
    rbinom(size = 1, prob = P[3], n = N[3]),
    rbinom(size = 1, prob = P[4], n = N[4])
  )
  
  simulation_data <- cbind(y, X, cluster) %>% 
    as.data.frame()
  colnames(simulation_data) <- c('y', 'x1', 'x2', 'cluster')
  simulation_data$y <- as.factor(simulation_data$y)
  return(simulation_data)
}

.sphere_data <- function(n, r_mean, r_sd) {
  r <- rnorm(n, r_mean, r_sd)
  theta <- runif(n, 0, 2*pi)
  x1 <- r * cos(theta)
  x2 <- r * sin(theta)
  return(data.frame(x1 = x1, x2 = x2))
}

sphere_data_gen <- function(ns, r_means, r_sds) {
  control <- .sphere_data(ns[1], r_means[1], r_sds[1])
  control$y <- 0
  case <- .sphere_data(ns[2], r_means[2], r_sds[2])
  case$y <- 1
  output <- rbind(case, control)
  output$y <- as.factor(output$y)
  output <- output[, c(3,1,2)]
  return(output)
}

get_logistic_decision_boundary <- function(lr, decision_rule=0.5) {
  z <- log(decision_rule / (1 - decision_rule))
  lr_coefs <- coef(lr)
  intercept <- (z - lr_coefs["(Intercept)"]) / lr_coefs["x2"]
  slope <- - lr_coefs["x1"] / lr_coefs["x2"]
  return(list(
    intercept=unname(intercept),
    slope=unname(slope)
  ))
}

prob_contour <- function(lr, lims) {
  grid_point <- seq(lims[1], lims[2], 0.05)
  grid_coord <- expand.grid(x1 = grid_point, x2 = grid_point)
  grid_coord <- cbind(
    grid_coord,
    p=predict(lr, newdata = grid_coord, type = "response")
  )
  return(grid_coord)
}

prob_contour_plot <- function(lr, lims, decision_rule=0.5) {
  decision_line <- get_logistic_decision_boundary(lr, decision_rule = decision_rule)
  ggplot() +
    geom_point(data = train, aes(x=x1, y=x2, color=y), alpha=0.5) +
    geom_raster(data = prob_contour(lr, lims = lims),
                aes(x = x1, y = x2, fill = p)) +
    scale_fill_viridis_c(alpha=0.5, lim=c(0, 1)) +
    geom_abline(slope = decision_line$slope, intercept = decision_line$intercept,
                linetype = "dotted", color = "red", linewidth = 1)
}

## simulation 1 ----------------------
# try on different sd
# large sd => well mixed clusters
# small sd => more isolated cluster
## --- --- --- --- --- --- --- --- --- --- --- --- --- 

sigma <- 0.2

train <- data_generator(sigma)
test <- data_generator(sigma)

# write.csv(train, "./train.csv", row.names = FALSE)
# write.csv(test, "./test.csv", row.names = FALSE)

p1 <- ggplot() +
  geom_point(data = train, aes(x=x1, y=x2, color=y), alpha=0.5) +
  lims(x = c(-1, 2), y = c(-1, 2))

k <- 1
p2 <- ggplot(data = train[train$cluster != k,],
       aes(x=x1, y=x2, color=y)) +
  geom_point(alpha=0.5) +
  lims(x = c(-1, 2), y = c(-1, 2))

lr <- glm(y ~ x1 + x2, family = binomial(), data = select(train, -cluster))
p <- predict(lr, test[,2:3], type = "response")
s1 <- auc(roc(test$y, p, quiet = TRUE))

lr_sub <- glm(
  y ~ x1 + x2, family = binomial(), 
  data = select(train[train$cluster != k,], -cluster)
)
p_sub <- predict(lr_sub, test[, 2:3], type = "response") %>% unname()
s2 <- auc(roc(test$y, p_sub, quiet = TRUE))
p_sub <- predict(lr_sub, test[test$cluster != 1, 2:3], type = "response") %>% unname()
s3 <- auc(roc(test$y, c(rep(0, 1000), p_sub), quiet = TRUE))

p1 + labs(title = glue::glue("baseline: ", round(s1, 3))) +
  p2 + labs(
    title = glue::glue(
      "downsampling: ",
      round(s2, 3)
    ),
    y = ""
  ) +
  # plot_annotation(title = glue::glue("Standard deviation: ", sigma, " [including the interaction term]")) +
  plot_annotation(
    title = glue::glue(
      "Visualizing training data: standard deviation = ",
      sigma,
      "& compute test AUC"
    )
  ) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom") &
  labs(color = "")


## simulation 2----------------------
# sphere data
## --- --- --- --- --- --- --- --- --- --- --- --- --- 

train <- sphere_data_gen(ns = c(1300, 200), r_means = c(5, 2), r_sds = c(1, 1))
test <- sphere_data_gen(ns = c(1300, 200), r_means = c(5, 2), r_sds = c(1, 1))

write.csv(train, "./simulate_data/sphere_train.csv", row.names = FALSE)
write.csv(test, "./simulate_data/sphere_test.csv", row.names = FALSE)

ggplot() +
  geom_point(data = train, aes(x = x1, y = x2, color = y))

lr <- glm(y ~ x1 + x2, family = binomial(), data = train)
p <- predict(lr, test[,c("x1", "x2")], type = "response")
auc(roc(test$y, p, quiet = TRUE))

prob_contour_plot(lr, c(-8, 8))
