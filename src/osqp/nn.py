import numpy as np
import scipy.sparse as spa
import torch
from torch.nn import Module
from torch.autograd import Function

import osqp


def to_numpy(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class OSQP(Module):
    def __init__(self, P_idx, P_shape, A_idx, A_shape,
                 eps_rel=1e-5, eps_abs=1e-5, verbose=False,
                 max_iter=10000):
        super().__init__()
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.verbose = verbose
        self.max_iter = max_iter
        self.P_idx, self.P_shape = P_idx, P_shape
        self.A_idx, self.A_shape = A_idx, A_shape

    def forward(self, P_val, q_val, A_val, l_val, u_val):
        return _OSQP.apply(
            P_val, q_val, A_val, l_val, u_val,
            self.P_idx, self.P_shape,
            self.A_idx, self.A_shape,
            self.eps_rel, self.eps_abs,
            self.verbose, self.max_iter
        )


class _OSQP(Function):
    @staticmethod
    def forward(ctx, P_val, q_val, A_val, l_val, u_val,
                P_idx, P_shape, A_idx, A_shape,
                eps_rel, eps_abs, verbose, max_iter):
        """Solve a batch of QPs using OSQP.

        This function solves a batch of QPs, each optimizing over
        `n` variables and having `m` constraints.

        The optimization problem for each instance in the batch
        (dropping indexing from the notation) is of the form

            \\hat x =   argmin_x 1/2 x' P x + q' x
                       subject to l <= Ax <= u

        where P \\in S^{n,n},
              S^{n,n} is the set of all positive semi-definite matrices,
              q \\in R^{n}
              A \\in R^{m,n}
              l \\in R^{m}
              u \\in R^{m}

        These parameters should all be passed to this function as
        Variable- or Parameter-wrapped Tensors.
        (See torch.autograd.Variable and torch.nn.parameter.Parameter)

        If you want to solve a batch of QPs where `n` and `m`
        are the same, but some of the contents differ across the
        minibatch, you can pass in tensors in the standard way
        where the first dimension indicates the batch example.
        This can be done with some or all of the coefficients.

        You do not need to add an extra dimension to coefficients
        that will not change across all of the minibatch examples.
        This function is able to infer such cases.

        If you don't want to use any constraints, you can set the
        appropriate values to:

            e = Variable(torch.Tensor())

        """

        ctx.eps_abs = eps_abs
        ctx.eps_rel = eps_rel
        ctx.verbose = verbose
        ctx.max_iter = max_iter
        ctx.P_idx, ctx.P_shape = P_idx, P_shape
        ctx.A_idx, ctx.A_shape = A_idx, A_shape

        params = [P_val, q_val, A_val, l_val, u_val]

        for p in params:
            assert p.ndimension() <= 2, 'Unexpected number of dimensions'

        # Convert batches to sparse matrices/vectors
        batch_mode = np.all([t.ndimension() == 1 for t in params])
        if batch_mode:
            ctx.n_batch = 1
        else:
            batch_sizes = [t.size(0) if t.ndimension() == 2 else 1 for t in params]
            ctx.n_batch = max(batch_sizes)
        ctx.m, ctx.n = ctx.A_shape   # Problem size

        dtype = P_val.dtype
        device = P_val.device

        # Convert P and A to sparse matrices
        # TODO (Bart): create CSC matrix during initialization. Then
        # just reassign the mat.data vector with A_val and P_val

        for i, p in enumerate(params):
            if p.ndimension() == 1:
                params[i] = p.unsqueeze(0).expand(ctx.n_batch, p.size(0))

        [P_val, q_val, A_val, l_val, u_val] = params

        P = [spa.csc_matrix((to_numpy(P_val[i]), ctx.P_idx), shape=ctx.P_shape)
             for i in range(ctx.n_batch)]
        q = [to_numpy(q_val[i]) for i in range(ctx.n_batch)]
        A = [spa.csc_matrix((to_numpy(A_val[i]), ctx.A_idx), shape=ctx.A_shape)
             for i in range(ctx.n_batch)]
        l = [to_numpy(l_val[i]) for i in range(ctx.n_batch)]
        u = [to_numpy(u_val[i]) for i in range(ctx.n_batch)]

        # Perform forward step solving the QPs
        x_torch = torch.zeros((ctx.n_batch, ctx.n), dtype=dtype, device=device)

        solvers = []
        for i in range(ctx.n_batch):
            # Solve QP
            m = osqp.OSQP()
            m.setup(P[i], q[i], A[i], l[i], u[i], verbose=ctx.verbose, scaling=0)
            result = m.solve()
            status = result.info.status
            if status != 'solved':
                # TODO: We can replace this with something calmer and
                # add some more options around potentially ignoring this.
                raise RuntimeError(f"Unable to solve QP, status: {status}")
            solvers.append(m)

            # This is silently converting result.x to the same
            # dtype and device as x_torch.
            x_torch[i] = torch.from_numpy(result.x)

        # Save solvers for backpropagation
        ctx.solvers = solvers

        # Return solutions
        if not batch_mode:
            x_torch = x_torch.squeeze(0)
        return x_torch

    @staticmethod
    def backward(ctx, dl_dx_val):
        dtype = dl_dx_val.dtype
        device = dl_dx_val.device

        batch_mode = dl_dx_val.ndimension() == 2
        if not batch_mode:
            dl_dx_val = dl_dx_val.unsqueeze(0)

        # Convert dl_dx to numpy
        dl_dx = to_numpy(dl_dx_val)

        # Extract data from forward pass
        solvers = ctx.solvers

        # Convert to torch tensors
        nnz_P = len(ctx.P_idx[0])
        nnz_A = len(ctx.A_idx[0])
        dP = torch.zeros((ctx.n_batch, nnz_P), dtype=dtype, device=device)
        dq = torch.zeros((ctx.n_batch, ctx.n), dtype=dtype, device=device)
        dA = torch.zeros((ctx.n_batch, nnz_A), dtype=dtype, device=device)
        dl = torch.zeros((ctx.n_batch, ctx.m), dtype=dtype, device=device)
        du = torch.zeros((ctx.n_batch, ctx.m), dtype=dtype, device=device)

        for i in range(ctx.n_batch):

            m = solvers[i]
            dP, dq, dA, dl, du = m.adjoint_derivative(dx=dl_dx[i])
            dP = torch.from_numpy(dP.data)
            dq = torch.from_numpy(dq)
            dA = torch.from_numpy(dA.data)
            dl = torch.from_numpy(dl)
            du = torch.from_numpy(du)

        grads = [dP, dq, dA, dl, du]

        if not batch_mode:
            for i, g in enumerate(grads):
                grads[i] = g.squeeze()

        grads += [None]*9

        return tuple(grads)