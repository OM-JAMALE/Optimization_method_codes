import numpy as np

def newton_minimize(grad, hess, x0, tol=1e-8, max_iter=100):
    x = np.array(x0, dtype=float)

    for i in range(max_iter):
        g = grad(x)                      # gradient at x
        if np.linalg.norm(g) < tol:      # gradient ~ 0 -> stop
            break

#   formula is Xi+1 = Xi - H^-1 * g
        H = hess(x)  # Hessian of f at x
        if np.linalg.det(H) == 0:
            return x,'Cant apply this method'
                               
        step = np.linalg.solve(H, g)     # euivalent to  H^-1 * g But solves it as H * step = g  
        x = x - step                     # Newton update

    return x,' Minima found !!!'


# Example: f(x0, x1) = x0^2 + x1^2 + exp(x0 + x1)
def f(x):
    return x[0]**3 + x[1]**2 

def grad(x):
    return np.array([3*x[0]**2, 2*x[1]])

def hess(x):
    e = np.exp(x[0] + x[1])
    return np.array([[6*x[0], 0],
                     [0,  2]])


print("\n")
x_min , note1 = newton_minimize(grad, hess, x0=[1.0, 1.0])
print(note1,"Minimum at:", x_min)
print("f(min) =", f(x_min),"\n")


# 2 ..... Example 2 with lambda functions for f, grad, and hess
f2    = lambda x: x[0]**2 + 10*x[1]**2
# gradient of f2
grad2 = lambda x: np.array([2*x[0], 20*x[1]])
hessian2 = lambda x: np.array([[2, 0], [0, 20]])

x_min2,note2 = newton_minimize(grad2, hessian2, x0=[5.0, 1.0])
print( note2,"Minimum at:", x_min2)
print("f(min) =", f2(x_min2),"\n")