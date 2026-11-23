import numpy as np
#           tol = 1e-6 defines a tolerance value of 0.000001
def steepest_descent(f, grad, hessian, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)

    for i in range(max_iter):
        g = grad(x)
        # np.linalg.norm(g) calculate norm of the gradient g 
        if np.linalg.norm(g) < tol:      # gradient ~ 0 -> stop the loop
            break

        p = -g                           # steepest descent direction

        # backtracking line search: shrink step until f decreases enough
        pT = np.transpose(p)
        Q = hessian(x)
        l = (pT @ p) / (pT @ Q @ p) # step size based on hessian
        x = x + l * p             # move

    return x


# approch 1 need to update the function and gradient for every new function we want to minimize
def f(x):
    return x[0]**3 + x[1]**2 

def grad(x):
    # returns gradient vector of f at x for every function we need to change the gradient function
    return np.array([3*(x[0]**2), 2*x[1]])

def hessian(x):
    # returns hessian matrix of f at x for every function we need to change the hessian function
    return np.array([[6*x[0], 0], [0, 2]])

print("\n")
print(steepest_descent(f, grad, hessian, [1.0, 1.0]))   # about [-0.28, -0.28]


# Example 2: stretched bowl f = x0^2 + 10*x1^2 (shows the slow zigzag)
f2    = lambda x: x[0]**2 + 10*x[1]**2
# gradient of f2
grad2 = lambda x: np.array([2*x[0], 20*x[1]])
hessian2 = lambda x: np.array([[2, 0], [0, 20]])

print(steepest_descent(f2, grad2, hessian2, [5.0, 1.0]))  # close to [0, 0]

print("\n")