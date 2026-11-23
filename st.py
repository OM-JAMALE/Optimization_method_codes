import numpy as np
import plotly.graph_objects as go


def plot_3d_path(f, path, title="3D Steepest Descent Path"):
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
        marker=dict(size=10, color="#ffea00", symbol="diamond", line=dict(color="black", width=2)),
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
    return x[0]**4 + x[1]**2


def grad(x):
    # returns gradient vector of f at x for every function we need to change the gradient function
    return np.array([4*(x[0]**3), 2*x[1]])


def hessian(x):
    # returns hessian matrix of f at x for every function we need to change the hessian function
    return np.array([[12*(x[0]**2), 0], [0, 2]])


print("\n")
print(steepest_descent(f, grad, hessian, [1.0, 1.0], title="f(x) = x0^3 + x1^2 (Zigzag) 3D Steepest descent method "))


# Example 2: stretched bowl f = x0^2 + 10*x1^2 (shows the slow zigzag)
f2 = lambda x: x[0]**2 + 10*x[1]**2
# gradient of f2
grad2 = lambda x: np.array([2*x[0], 20*x[1]])
hessian2 = lambda x: np.array([[2, 0], [0, 20]])

print(steepest_descent(f2, grad2, hessian2, [5.0, 1.0], title="f2(x) = x0^2 + 10*x1^2 (Zigzag) 3D Steepest descent method"))  # close to [0, 0]

print("\n")




















# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm
# from matplotlib.colors import LightSource, Normalize


# def plot_3d_path(f, path, title="3D Steepest Descent Path"):
#     path = np.array(path)
#     z_path = np.array([f(pt) for pt in path])

#     # Build a grid around the visited points so the surface frames the path clearly
#     x_min, x_max = path[:, 0].min(), path[:, 0].max()
#     y_min, y_max = path[:, 1].min(), path[:, 1].max()
#     pad_x = max((x_max - x_min) * 0.25, 1.0)
#     pad_y = max((y_max - y_min) * 0.25, 1.0)

#     # Denser grid -> smoother, more realistic surface
#     x_grid = np.linspace(x_min - pad_x, x_max + pad_x, 150)
#     y_grid = np.linspace(y_min - pad_y, y_max + pad_y, 150)
#     X, Y = np.meshgrid(x_grid, y_grid)
#     Z = f([X, Y])

#     # Vibrant colormap + simulated light source for realistic depth/shading
#     cmap = cm.turbo
#     norm = Normalize(vmin=Z.min(), vmax=Z.max())
#     ls = LightSource(azdeg=315, altdeg=45)
#     face_colors = ls.shade(Z, cmap=cmap, norm=norm, vert_exag=0.6, blend_mode="soft")

#     fig = plt.figure(figsize=(11, 7.5), facecolor="#0b0f1a")
#     ax = fig.add_subplot(111, projection="3d", facecolor="#0b0f1a")

#     ax.plot_surface(
#         X, Y, Z,
#         facecolors=face_colors,
#         rstride=1, cstride=1,
#         linewidth=0,
#         antialiased=True,
#         shade=False,
#         alpha=0.92,
#     )

#     # Colored contour "shadow" projected on the floor for extra depth
#     z_floor = Z.min() - 0.08 * (Z.max() - Z.min())
#     ax.contourf(X, Y, Z, zdir="z", offset=z_floor, levels=25, cmap=cmap, alpha=0.75)
#     ax.contour(X, Y, Z, zdir="z", offset=z_floor, levels=25, colors="white", linewidths=0.3, alpha=0.4)

#     # Small lift so the path is not hidden inside the surface (visual only)
#     lift = 0.01 * (Z.max() - Z.min())
#     zp = z_path + lift

#     # Glow effect: wide soft line under a bright thin line
#     ax.plot(path[:, 0], path[:, 1], zp, color="white", linewidth=6, alpha=0.25)
#     ax.plot(
#         path[:, 0], path[:, 1], zp,
#         color="#ff1744",
#         marker="o",
#         markersize=5,
#         markeredgecolor="white",
#         markeredgewidth=0.6,
#         linewidth=2.5,
#         label="Iterates",
#     )
#     ax.scatter(
#         path[0, 0], path[0, 1], zp[0],
#         color="#00e5ff", edgecolor="white", s=90, depthshade=False, label="Start",
#     )
#     ax.scatter(
#         path[-1, 0], path[-1, 1], zp[-1],
#         color="#ffea00", edgecolor="black", marker="*", s=220, depthshade=False, label="Final point",
#     )

#     # Dark-theme styling
#     ax.set_title(title, color="white", fontsize=14, pad=12)
#     for axis_label, setter in (("x0", ax.set_xlabel), ("x1", ax.set_ylabel), ("f(x0, x1)", ax.set_zlabel)):
#         setter(axis_label, color="white")
#     ax.tick_params(colors="white")
#     for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
#         pane.set_pane_color((0.05, 0.07, 0.12, 0.0))
#         pane._axinfo["grid"]["color"] = (1, 1, 1, 0.12)
#     ax.set_zlim(z_floor, Z.max())
#     ax.view_init(elev=28, azim=-55)

#     mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
#     mappable.set_array([])
#     cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08)
#     cbar.set_label("f value", color="white")
#     cbar.ax.yaxis.set_tick_params(color="white")
#     plt.setp(cbar.ax.get_yticklabels(), color="white")

#     leg = ax.legend(facecolor="#1a2035", edgecolor="none")
#     for t in leg.get_texts():
#         t.set_color("white")

#     plt.tight_layout()
#     plt.show()


# # tol = 1e-6 defines a tolerance value of 0.000001
# def steepest_descent(f, grad, hessian, x0, tol=1e-6, max_iter=1000, plot=True, title="3D Steepest Descent Path"):
#     x = np.array(x0, dtype=float)
#     path = [x.copy()]

#     for i in range(max_iter):
#         g = grad(x)
#         # np.linalg.norm(g) calculate norm of the gradient g
#         if np.linalg.norm(g) < tol:      # gradient ~ 0 -> stop the loop
#             break

#         p = -g                           # steepest descent direction

#         pT = np.transpose(p)
#         Q = hessian(x)
#         denom = pT @ Q @ p
#         if denom <= 0:                   # guard against non-positive definite Hessian (e.g., on x0^3 + x1^2)
#             break
#         l = (pT @ p) / denom             # step size based on hessian
#         x = x + l * p                    # move
#         path.append(x.copy())

#     if plot:
#         plot_3d_path(f, path, title=title)

#     return x


# # approch 1 need to update the function and gradient for every new function we want to minimize
# def f(x):
#     return x[0]**3 + x[1]**2


# def grad(x):
#     # returns gradient vector of f at x for every function we need to change the gradient function
#     return np.array([3*(x[0]**2), 2*x[1]])


# def hessian(x):
#     # returns hessian matrix of f at x for every function we need to change the hessian function
#     return np.array([[6*x[0], 0], [0, 2]])


# print("\n")
# print(steepest_descent(f, grad, hessian, [1.0, 1.0], title="f(x) = x0^3 + x1^2"))


# # Example 2: stretched bowl f = x0^2 + 10*x1^2 (shows the slow zigzag)
# f2 = lambda x: x[0]**2 + 10*x[1]**2
# # gradient of f2
# grad2 = lambda x: np.array([2*x[0], 20*x[1]])
# hessian2 = lambda x: np.array([[2, 0], [0, 20]])

# print(steepest_descent(f2, grad2, hessian2, [5.0, 1.0], title="f2(x) = x0^2 + 10*x1^2 (Zigzag)"))  # close to [0, 0]

# print("\n")
















# import matplotlib.pyplot as plt
# import numpy as np


# def plot_3d_path(f, path, title="3D Steepest Descent Path"):
#     path = np.array(path)
#     z_path = np.array([f(pt) for pt in path])

#     # Build a grid around the visited points so the surface frames the path clearly
#     x_min, x_max = path[:, 0].min(), path[:, 0].max()
#     y_min, y_max = path[:, 1].min(), path[:, 1].max()
#     pad_x = max((x_max - x_min) * 0.25, 1.0)
#     pad_y = max((y_max - y_min) * 0.25, 1.0)

#     x_grid = np.linspace(x_min - pad_x, x_max + pad_x, 60)
#     y_grid = np.linspace(y_min - pad_y, y_max + pad_y, 60)
#     X, Y = np.meshgrid(x_grid, y_grid)
#     Z = f([X, Y])

#     fig = plt.figure(figsize=(9, 6))
#     ax = fig.add_subplot(111, projection="3d")

#     ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.55, edgecolor="none")
#     ax.plot(
#         path[:, 0],
#         path[:, 1],
#         z_path,
#         color="red",
#         marker="o",
#         markersize=4,
#         linewidth=2,
#         label="Iterates",
#     )
#     ax.scatter(
#         path[0, 0],
#         path[0, 1],
#         z_path[0],
#         color="blue",
#         s=60,
#         label="Start",
#     )
#     ax.scatter(
#         path[-1, 0],
#         path[-1, 1],
#         z_path[-1],
#         color="black",
#         marker="*",
#         s=120,
#         label="Final point",
#     )

#     ax.set_title(title)
#     ax.set_xlabel("x0")
#     ax.set_ylabel("x1")
#     ax.set_zlabel("f(x0, x1)")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()


# # tol = 1e-6 defines a tolerance value of 0.000001
# def steepest_descent(f, grad, hessian, x0, tol=1e-6, max_iter=1000, plot=True, title="3D Steepest Descent Path"):
#     x = np.array(x0, dtype=float)
#     path = [x.copy()]

#     for i in range(max_iter):
#         g = grad(x)
#         # np.linalg.norm(g) calculate norm of the gradient g
#         if np.linalg.norm(g) < tol:      # gradient ~ 0 -> stop the loop
#             break

#         p = -g                           # steepest descent direction

#         pT = np.transpose(p)
#         Q = hessian(x)
#         denom = pT @ Q @ p
#         if denom <= 0:                   # guard against non-positive definite Hessian (e.g., on x0^3 + x1^2)
#             break
#         l = (pT @ p) / denom             # step size based on hessian
#         x = x + l * p                    # move
#         path.append(x.copy())

#     if plot:
#         plot_3d_path(f, path, title=title)

#     return x


# # approch 1 need to update the function and gradient for every new function we want to minimize
# def f(x):
#     return x[0]**3 + x[1]**2


# def grad(x):
#     # returns gradient vector of f at x for every function we need to change the gradient function
#     return np.array([3*(x[0]**2), 2*x[1]])


# def hessian(x):
#     # returns hessian matrix of f at x for every function we need to change the hessian function
#     return np.array([[6*x[0], 0], [0, 2]])


# print("\n")
# print(steepest_descent(f, grad, hessian, [1.0, 1.0], title="f(x) = x0^3 + x1^2"))


# # Example 2: stretched bowl f = x0^2 + 10*x1^2 (shows the slow zigzag)
# f2 = lambda x: x[0]**2 + 10*x[1]**2
# # gradient of f2
# grad2 = lambda x: np.array([2*x[0], 20*x[1]])
# hessian2 = lambda x: np.array([[2, 0], [0, 20]])

# print(steepest_descent(f2, grad2, hessian2, [5.0, 1.0], title="f2(x) = x0^2 + 10*x1^2 (Zigzag)"))  # close to [0, 0]

# print("\n")