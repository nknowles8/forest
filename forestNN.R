nnComputeSingle <- function(X, theta, hidden_size){
      
      m <- nrow(X)
      input_size <- ncol(X)
      #output_size <- (length(theta) - (input_size + 1) * hidden_size)/(hidden_size + 1)
      
      theta1 <- matrix(theta[1:((input_size + 1) * hidden_size)], input_size + 1, hidden_size)
      theta2 <- matrix(theta[-(1:((input_size + 1) * hidden_size))], hidden_size + 1)
      
      a1 <- cbind(rep(1, m), X)
      
      z2 <- a1 %*% theta1
      a2 <- cbind(rep(1, m), sigmoid(z2))
      
      z3 <- a2 %*% theta2
      h <- sigmoid(z3)
      
      return(h)
      
}

nnTrainSingle <- function(X, y, theta_init, hidden_size = 1, lambda = 0, learning_rate = 0.01, threshold = 0.0001, max_iter = 400){
#this function trains a single hidden layer neural network to the data and outputs a rolled theta vector
      
      J_vec <- NULL
      
      all_theta <- theta_init
      
      for (i in 1:max_iter){
            
            net <- nnCostFnSingle(X, y, all_theta, hidden_size, lambda)
            J_vec[i] <- net$cost
            all_theta <- all_theta - learning_rate * net$gradient
            
            print(length(J_vec))
            
            if(i >= 10){
                  if(J_vec[length(J_vec) - 1] - J_vec[length(J_vec)] < threshold){
                        
                        return(list(theta = all_theta, cost = J_vec))
                        
                  }
                  
            }
      }
      
      cat('Maximum iteration threshold reached')
      return(list(theta = all_theta, cost = J_vec))
}

nnCostFnSingle <- function(X, y, all_theta, hidden_size = 1, lambda = 0){
#this function computes the cost and gradient for a single hidden layer neural network
#X is an explanatory variable matrix (mxn) with m observations and n features
#y is a target variable matrix (mxp) with m observations and p output variables
#theta is a vector containing the two "rolled" theta matrices
#hidden_size specifies the size of the hidden layer
#lambda is a regularization constant with a default value of 0
      
      m <- nrow(X)
      input_size <- ncol(X)
      output_size <- ncol(y)
      
      #unroll all_theta
      theta1 <- matrix(all_theta[1:((input_size + 1) * hidden_size)], nrow = input_size + 1, ncol = hidden_size)
      theta2 <- matrix(all_theta[-(1:((input_size + 1) * hidden_size))], nrow = hidden_size + 1, ncol = output_size)
      
      #forward propogation
      a1 <- cbind(rep(1, m), X)
      
      z2 <- a1 %*% theta1
      a2 <- cbind(rep(1, m), sigmoid(z2))
      
      z3 <- a2 %*% theta2
      h <- sigmoid(z3)
      
      J <- (-1/m) * sum(y * log(h) + (1 - y) * log(1 - h)) + lambda/(2 * m) * sum(all_theta^2)
      
      #back prop
      del3 <- h - y
      del2 <- del3 %*% t(theta2) * ((a2) * (1 - a2))
                                    
      Delta2 <- t(a2) %*% del3
      Delta1 <- t(a1) %*% del2[, -1]
      
      all_Delta <- c(as.vector(c(Delta1)), as.vector(c(Delta2)))
      
      return(list(cost = J, gradient = all_Delta))
      
}

sigmoid <- function(z){
      
      return(1/(1 + exp(-z)))
      
}