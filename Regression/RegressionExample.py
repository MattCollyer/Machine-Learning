
import math


def gradient_descent(w0, w1, X, Y, step_size, tolerance, max_iterations):
  derivative_w0 = 0
  derivative_w1 = 0
  rss = 0
  for i in range(0,len(X)):
    y_hat = w0 + X[i] * w1
    error = (y_hat - Y[i])
    derivative_w0 += error
    derivative_w1 += error * X[i]
    rss += error**2 
  w0 = w0 - derivative_w0 * step_size
  w1 = w1 - derivative_w1 * step_size 


  if((max_iterations < 0) or (math.sqrt(derivative_w0**2 + derivative_w1**2) < tolerance)):
    return([w0, w1, rss, 1001-max_iterations])
  else:
    return gradient_descent(w0, w1, X, Y, step_size, tolerance, max_iterations-1)
    
    



X = [0, 1, 2, 3, 4]

Y = [1, 2, 5, 10, 17]


answer = gradient_descent(0, 0, X, Y, 0.05, 0.01, 1000)
print(answer)

print(answer[1]*2.5+answer[0])
