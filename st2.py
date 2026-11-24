import numpy as np
import plotly.graph_objects as go


def plot_3d_path(f, path, title="3D Steepest Descent Path"):
    path = np.array(path)
    z_path = np.array([f(pt) for pt in path])

    # Center the grid on the FINAL point and size it so the START point (and the
    # whole path) fits inside. Change MARGIN to zoom: smaller = tighter/bigger path.
    MARGIN = 1.15
    final = path[-1]
    half_x = max(np.abs(path[:, 0] - final[0]).max() * MARGIN, 1.0)
    half_y = max(np.abs(path[:, 1] - final[1]).max() * MARGIN, 1.0)

    x_grid = np.linspace(final[0] - half_x, final[0] + half_x, 120)
    y_grid = np.linspace(final[1] - half_y, final[1] + half_y, 120)
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
        marker=dict(size=10, color="#ffea00", symbol="diamond", line=dict(color="black", width=2)),
        name="Final point",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18)),
        paper_bgcolor="#0b0f1a",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a2035"),
        height=900,                      # bigger figure (pixels)
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title="x0", backgroundcolor="#0b0f1a", gridcolor="#2a3150", color="white"),
            yaxis=dict(title="x1", backgroundcolor="#0b0f1a", gridcolor="#2a3150", color="white"),
            zaxis=dict(title="f(x0, x1)", backgroundcolor="#0b0f1a", gridcolor="#2a3150", color="white",
                       range=[z_floor, Z.max()]),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.8)),   # smaller numbers = camera closer = bigger graph
        ),
    )
    # Opens in your browser; rendering is done with WebGL on the GPU
    fig.show()


# tol = 1e-6 defines a tolerance value of 0.000001
def steepest_descent(f, grad, hessian, x0, tol=1e-6, max_iter=1000, plot=True, title="3D Steepest Descent Path"):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for i in range(max_iter):
        g = grad(x)
        # np.linalg.norm(g) calculate norm of the gradient g
        if np.linalg.norm(g) < tol:      # gradient ~ 0 -> stop the loop
            break

        p = -g                           # steepest descent direction

        pT = np.transpose(p)
        Q = hessian(x)
        denom = pT @ Q @ p
        if denom <= 0:                   # guard against non-positive definite Hessian (e.g., on x0^3 + x1^2)
            break
        l = (pT @ p) / denom             # step size based on hessian
        x = x + l * p                    # move
        path.append(x.copy())

    if plot:
        plot_3d_path(f, path, title=title)

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
print(steepest_descent(f, grad, hessian, [1.0, 1.0], title="f(x) = x0^3 + x1^2"))


# Example 2: stretched bowl f = x0^2 + 10*x1^2 (shows the slow zigzag)
f2 = lambda x: x[0]**2 + 10*x[1]**2
# gradient of f2
grad2 = lambda x: np.array([2*x[0], 20*x[1]])
hessian2 = lambda x: np.array([[2, 0], [0, 20]])

print(steepest_descent(f2, grad2, hessian2, [5.0, 1.0], title="f2(x) = x0^2 + 10*x1^2 (Zigzag)"))  # close to [0, 0]

print("\n")