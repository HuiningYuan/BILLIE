from build_library import PolyDiffPoint, build_Theta, build_linear_system
import numpy as np


def deleteDuplicatedElementFromList(input):
    # return list(set(listA))
    return sorted(list(set(input)))


# get weight
def get_weight(PDE_tgt, PDE_library, candidates):
    candidates = deleteDuplicatedElementFromList(candidates)
    truncated_PDE_library = PDE_library[:, candidates]
    weight = np.matmul(
        np.matmul(np.linalg.pinv(np.matmul(truncated_PDE_library.transpose(1, 0), truncated_PDE_library)),
                  truncated_PDE_library.transpose(1, 0)), PDE_tgt)
    return candidates, weight


# reward function
def get_reward(tgt_train, library_train, candidates, tgt_val, library_val, a1=1e4, a2=1e4, a3=1):
    # sort
    candidates = deleteDuplicatedElementFromList(candidates)
    # get l0 regularization
    l0_loss = len(candidates)
    # get weight
    truncated_library_train = library_train[:, candidates]
    weight = np.matmul(
        np.matmul(np.linalg.pinv(np.matmul(truncated_library_train.transpose(1, 0), truncated_library_train)),
                  truncated_library_train.transpose(1, 0)), tgt_train)
    l2_loss_train = (np.square(np.subtract(np.matmul(truncated_library_train, weight), tgt_train))).mean()
    # validation
    truncated_library_val = library_val[:, candidates]
    l2_loss_val = (np.square(np.subtract(np.matmul(truncated_library_val, weight), tgt_val))).mean()
    # get reward
    reward = -a1 * l2_loss_train - a2 * l2_loss_val - a3 * l0_loss
    reward = -np.log(-reward)
    return reward, l2_loss_train, l2_loss_val, l0_loss


# reward function (log)
def get_reward_log(tgt_train, library_train, candidates, tgt_val, library_val, a1=1e4, a2=1e4, a3=1):
    # sort
    candidates = deleteDuplicatedElementFromList(candidates)
    # get l0 regularization
    l0_loss = len(candidates)
    # get weight
    truncated_library_train = library_train[:, candidates]
    weight = np.matmul(
        np.matmul(np.linalg.pinv(np.matmul(truncated_library_train.transpose(1, 0), truncated_library_train)),
                  truncated_library_train.transpose(1, 0)), tgt_train)
    l2_loss_train = (np.square(np.subtract(np.matmul(truncated_library_train, weight), tgt_train))).mean()
    # validation
    truncated_library_val = library_val[:, candidates]
    l2_loss_val = (np.square(np.subtract(np.matmul(truncated_library_val, weight), tgt_val))).mean()
    # get reward
    reward = -a1 * l2_loss_train - a2 * l2_loss_val - a3 * l0_loss
    reward = -np.log(-reward)
    return reward, l2_loss_train, l2_loss_val, l0_loss


# Burgers equation structured data processing
def build_burgers_data(u, dx, dt, D, P, noise_level=0.0, deg_x=4, deg_t=4, width_x=10, width_t=10):
    if noise_level == 0.0:
        time_diff = 'poly'
        space_diff = 'poly'
        u = u + noise_level * np.std(u) * np.random.randn(u.shape[0], u.shape[1])
        Ut, Q, description = build_linear_system(u, dt, dx, D=D, P=P, time_diff=time_diff,
                                                 space_diff=space_diff, deg_t=4, deg_x=4,
                                                 width_t=10, width_x=10)
    else:
        time_diff = 'FD'
        space_diff = 'FD'
        Ut, Q, description = build_linear_system(u, dt, dx, D=D, P=P, time_diff=time_diff,
                                                 space_diff=space_diff)
    Q = np.real(Q)
    Ut = [np.real(Ut)]
    return Ut, Q, description


# 2D Navier-Stokes equation structured data processing
def build_ns_data(U, V, P, points, num_points=10000, dx=2 * np.pi / 256, dy=2 * np.pi / 256, dt=0.0015,
                  boundary_x=10, boundary_y=10, boundary_t=10, deg=5, D=1):
    u = np.zeros((num_points, 1))
    v = np.zeros((num_points, 1))
    p = np.zeros((num_points, 1))
    # temporal derivative
    ut = np.zeros((num_points, 1))
    vt = np.zeros((num_points, 1))
    # first-order spatial derivative
    ux = np.zeros((num_points, 1))
    uy = np.zeros((num_points, 1))
    vx = np.zeros((num_points, 1))
    vy = np.zeros((num_points, 1))
    px = np.zeros((num_points, 1))
    py = np.zeros((num_points, 1))
    # second-order spatial derivative
    uxx = np.zeros((num_points, 1))
    uxy = np.zeros((num_points, 1))
    uyy = np.zeros((num_points, 1))
    vxx = np.zeros((num_points, 1))
    vxy = np.zeros((num_points, 1))
    vyy = np.zeros((num_points, 1))
    pxx = np.zeros((num_points, 1))
    pxy = np.zeros((num_points, 1))
    pyy = np.zeros((num_points, 1))
    Nx = 2 * boundary_x + 1
    Ny = 2 * boundary_y + 1
    Nt = 2 * boundary_t + 1
    for i in points.keys():
        [x, y, t] = points[i]
        u[i] = U[x, y, t]
        v[i] = V[x, y, t]
        p[i] = P[x, y, t]
        # process u
        # temporal derivative
        ut[i] = PolyDiffPoint(U[x, y, int(t - (Nt - 1) / 2): int(t + (Nt - 1) / 2 + 1)], np.arange(Nt) * dt, deg, 1)[0]
        # x derivative
        ux_diff = PolyDiffPoint(U[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y, t], np.arange(Nx) * dx, deg, 2)
        ux[i] = ux_diff[0]
        uxx[i] = ux_diff[1]
        # y derivative
        uy_diff = PolyDiffPoint(U[x, int(y - (Ny - 1) / 2):int(y + (Ny - 1) / 2 + 1), t], np.arange(Ny) * dy, deg, 2)
        uy[i] = uy_diff[0]
        uyy[i] = uy_diff[1]
        # mix derivative
        ux_diff_yp = PolyDiffPoint(U[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y + 1, t], np.arange(Nx) * dx,
                                   deg, 2)
        ux_diff_ym = PolyDiffPoint(U[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y - 1, t], np.arange(Nx) * dx,
                                   deg, 2)
        uxy[i] = (ux_diff_yp[0] - ux_diff_ym[0]) / (2 * dy)
        # process v
        # temporal derivative
        vt[i] = PolyDiffPoint(V[x, y, int(t - (Nt - 1) / 2):int(t + (Nt - 1) / 2 + 1)], np.arange(Nt) * dt, deg, 1)[0]
        # x derivative
        vx_diff = PolyDiffPoint(V[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y, t], np.arange(Nx) * dx, deg, 2)
        vx[i] = vx_diff[0]
        vxx[i] = vx_diff[1]
        # y derivative
        vy_diff = PolyDiffPoint(V[x, int(y - (Ny - 1) / 2):int(y + (Ny - 1) / 2 + 1), t], np.arange(Ny) * dy, deg, 2)
        vy[i] = vy_diff[0]
        vyy[i] = vy_diff[1]
        # mix derivative
        vx_diff_yp = PolyDiffPoint(V[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y + 1, t], np.arange(Nx) * dx,
                                   deg, 2)
        vx_diff_ym = PolyDiffPoint(V[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y - 1, t], np.arange(Nx) * dx,
                                   deg, 2)
        vxy[i] = (vx_diff_yp[0] - vx_diff_ym[0]) / (2 * dy)
        # process p
        # x derivative
        px_diff = PolyDiffPoint(P[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y, t], np.arange(Nx) * dx, deg, 2)
        px[i] = px_diff[0]
        pxx[i] = px_diff[1]
        # y derivative
        py_diff = PolyDiffPoint(P[x, int(y - (Ny - 1) / 2):int(y + (Ny - 1) / 2 + 1), t], np.arange(Ny) * dy, deg, 2)
        py[i] = py_diff[0]
        pyy[i] = py_diff[1]
        # mix derivative
        px_diff_yp = PolyDiffPoint(P[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y + 1, t], np.arange(Nx) * dx,
                                   deg, 2)
        px_diff_ym = PolyDiffPoint(P[int(x - (Nx - 1) / 2):int(x + (Nx - 1) / 2 + 1), y - 1, t], np.arange(Nx) * dx,
                                   deg, 2)
        pxy[i] = (px_diff_yp[0] - px_diff_ym[0]) / (2 * dy)
        print(i, 'done')
    # Form a huge matrix using up to quadratic polynomials in all variables.
    X_data = np.hstack([u, v, p])
    X_ders = np.hstack([np.ones((num_points, 1)), ux, uy, uxx, uxy, uyy, vx, vy, vxx, vxy, vyy, px, py, pxx, pxy, pyy])
    X_ders_descr = ['', 'u_{x}', 'u_{y}', 'u_{xx}', 'u_{xy}', 'u_{yy}', 'v_{x}', 'v_{y}', 'v_{xx}', 'v_{xy}', 'v_{yy}',
                    'p_{x}', 'p_{y}', 'p_{xx}', 'p_{xy}', 'p_{yy}']
    library, description = build_Theta(X_data, X_ders, X_ders_descr, D, data_description=['u', 'v', 'p'])
    return [ut, vt], np.real(library), description
