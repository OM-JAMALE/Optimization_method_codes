import numpy as np
import plotly.graph_objects as go


def plot_3d_path(f, path, title="3D Newton raphson method"):
    path = np.array(path)
    z_path = np.array([f(pt) for pt in path])

    # Build a grid around the visited points so the surface frames the path clearly
    x_min, x_max = path[:, 0].min(), path[:, 0].max()
    y_min, y_max = path[:, 1].min(), path[:, 1].max()
    pad_x = max((x_max - x_min) * 0.25, 1.0)
    pad_y = max((y_max - y_min) * 0.25, 1.0)

    x_grid = np.linspace(x_min - pad_x, x_max + pad_x, 120)
    y_grid = np.linspace(y_min - pad_y, y_max + pad_y, 120)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f([X, Y])

    z_range = Z.max() - Z.min()
    z_floor = Z.min() - 0.08 * z_range
    lift = 0.01 * z_range  # visual only, keeps the path from sinking into the surface
    zp = z_path + lift

    fig = go.Figure()

    # Vibrant, lit surface with a colored contour map projected on the floor
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Turbo",
        opacity=0.95,
        lighting=dict(ambient=0.45, diffuse=0.9, specular=0.6, roughness=0.4, fresnel=0.3),
        lightposition=dict(x=-200, y=-200, z=300),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        colorbar=dict(title="f value", len=0.7),
        name="Surface",
    ))

    # Glow under the path, then the bright path itself
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=zp, mode="lines",
        line=dict(color="white", width=12), opacity=0.3, showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=zp, mode="lines+markers",
        line=dict(color="#ff1744", width=6),
        marker=dict(size=4, color="#ff1744", line=dict(color="white", width=1)),
        name="Iterates",
    ))
    fig.add_trace(go.Scatter3d(
        x=[path[0, 0]], y=[path[0, 1]], z=[zp[0]], mode="markers",
        marker=dict(size=8, color="#00e5ff", line=dict(color="white", width=2)),
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[path[-1, 0]], y=[path[-1, 1]], z=[zp[-1]], mode="markers",
        marker=dict(size=10, color="#09fa1d", symbol="diamond", line=dict(color="black", width=2)),
        name="Final point",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18)),
        paper_bgcolor="#0b0f1a",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a2035"),
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title="x0", backgroundcolor="#0b0f1a", gridcolor="#2a3150", color="white"),
            yaxis=dict(title="x1", backgroundcolor="#0b0f1a", gridcolor="#2a3150", color="white"),
            zaxis=dict(title="f(x0, x1)", backgroundcolor="#0b0f1a", gridcolor="#2a3150", color="white",
                       range=[z_floor, Z.max()]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
    # Opens in your browser; rendering is done with WebGL on the GPU
    fig.show()


def newton_minimize(f,grad, hess, x0, tol=1e-8, max_iter=100,plot=True, title="3D Newton raphson method"):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
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
        path.append(x.copy())

    if plot:
            plot_3d_path(f, path, title=title)     

    return x,' Minima found !!!'


def f(x):
    return x[0]**4 + x[1]**2


def grad(x):
    # returns gradient vector of f at x for every function we need to change the gradient function
    return np.array([4*(x[0]**3), 2*x[1]])


def hessian(x):
    # returns hessian matrix of f at x for every function we need to change the hessian function
    return np.array([[12*(x[0]**2), 0], [0, 2]])

print("\n")
print(newton_minimize(f, grad, hessian, [1.0, 1.0], title="f(x) = x0^3 + x1^2 Newton raphson method"))


f2 = lambda x: x[0]**2 + 10*x[1]**2
# gradient of f2
grad2 = lambda x: np.array([2*x[0], 20*x[1]])
hessian2 = lambda x: np.array([[2, 0], [0, 20]])
print(newton_minimize(f2,grad2,hessian2,[5.0,1.0],title = " Newton raphson method 3d visualise "))