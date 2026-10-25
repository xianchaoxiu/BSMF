import time

import torch
import numpy as np


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def find_other2(A, W, nnz, Z, U, print_sc=None, debug=False, reg=0, rho_start=0.03, iters=5, prune_iters=2, fixmask=None):
    XX = A.T.matmul(A)
    norm2 = torch.diag(XX).sqrt() + 1e-8
    An = A / norm2
    XX = An.T.matmul(An)
    XX += torch.diag(torch.ones_like(XX.diag())) * XX.diag().mean() * reg

    Wnn = W
    rho = 1
    XY = An.T.matmul(Wnn)
    XXinv = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device)*rho)
    XXinv2 = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device)*rho_start)
    U = U * norm2.unsqueeze(1)
    Z = Z * norm2.unsqueeze(1)

    B = XXinv2.matmul(XY + rho_start*(Z-U))


    bsparsity = min(0.99, 1 - nnz/B.numel())


    for itt in range(iters):
        if itt < prune_iters and fixmask is None:
            cur_sparsity = bsparsity
            thres = (B+U).abs().flatten().sort()[0][int(B.numel() * cur_sparsity)]
            mask = ((B+U).abs() > thres)
            del thres
        if fixmask is not None:
            assert fixmask.shape == Z.shape
            mask = fixmask

        Z = (B + U) * mask

        U = U + (B - Z)

        B = XXinv.matmul(XY + rho*(Z-U))
        if debug:
            print(itt, cur_sparsity, (Z != 0).sum().item() / Z.numel())
            print(((An != 0).sum() + (Z != 0).sum()) / W.numel())
            print("-------")
    if debug:
        print("opt end")

    return Z / norm2.unsqueeze(1), U / norm2.unsqueeze(1)

def mag_prune(W, sp=0.6):
    thres = (W).abs().flatten().sort()[0][int(W.numel() * sp)]
    mask = ((W).abs() > thres)
    return W * mask

def ent(p):
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

def factorizeT(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    if fixmask is None:
        nza = int(W.shape[0]**2 * asp)
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)

    Az = torch.eye(W.shape[0], device=W.device)
    Au = torch.zeros_like(Az)
    norm = XX.diag().sqrt().unsqueeze(1) + 1e-8

    Wn = W * norm

    Bz = mag_prune(Wn, (1 - nzb/2/W.numel()))
    Bu = torch.zeros_like(Bz)

    for itt in range(iters):
        rho_start = min(1.0, itt / (iters-3))**3
        Az, Au = (x.T for x in find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask))

        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, debug=False, rho_start=rho_start)

    print(((Az != 0).sum() + (Bz != 0).sum()).item() / W.numel(), (Az != 0).sum().item() / Az.numel(),
          (Bz != 0).sum().item() / Bz.numel(), Az.shape, Bz.shape,
         (Az.numel()*ent((Az != 0).sum().item() / Az.numel()) + Bz.numel()*ent((Bz != 0).sum().item() / Bz.numel())) / W.numel(),
        ent(0.4), ent(0.5))
    return ((Az / norm).matmul(Bz)).T, Bz.T, (Az / norm).T


def factorizef(W, XX, asp=0.16, sp=0.4, iters=40, fixmask=None):
    s_time = time.time()
    if W.shape[0] >= W.shape[1]:###“行数 ≥ 列数”
        return factorizeT(W.T, XX, asp, sp=sp, fixmask=fixmask)

    if fixmask is None:
        nza = int(W.shape[0]**2 * asp)
    else:
        nza = (fixmask != 0).sum().item()
    nzb = int(W.numel() * sp - nza)
    norm = XX.diag().sqrt() + 1e-8

    Wn = W * norm

    Az = torch.eye(W.shape[0], device=W.device)###单位矩阵A
    Au = torch.zeros_like(Az)####A对应的U

    Bz = mag_prune(Wn, (1 - nzb/2/W.numel()))##初始化B
    Bu = torch.zeros_like(Bz)###B 对应的 U

    for itt in range(iters):

        rho_start = min(1.0, itt / (iters-3))**3##初试ρ值
        Az, Au = (x.T for x in find_other2(Bz.T, Wn.T, nza, Az.T, Au.T, reg=1e-2, debug=False, rho_start=rho_start, fixmask=fixmask))##固定B，更新A
        Bz, Bu = find_other2(Az, Wn, nzb, Bz, Bu, reg=1e-2, debug=False, rho_start=rho_start)###固定A更新B


    print(((Az != 0).sum() + (Bz != 0).sum()).item() / W.numel(), (Az != 0).sum().item() / Az.numel(),
          (Bz != 0).sum().item() / Bz.numel(), Az.shape, Bz.shape,
         (Az.numel()*ent((Az != 0).sum().item() / Az.numel()) + Bz.numel()*ent((Bz != 0).sum().item() / Bz.numel())) / W.numel(),
        ent(0.4), ent(0.5))
    return Az.matmul(Bz / norm), Az, Bz / norm

def finalize(XXb, W, Ab, Bb):
    fsparsity = 1 - (Ab != 0).sum() / Ab.numel()
    mask = (Ab != 0).T

    XX = Bb.matmul(XXb).matmul(Bb.T)
    XY = Bb.matmul(XXb).matmul(W.T)

    norm2 = torch.diag(XX).sqrt() + 1e-8
    XX = XX / norm2 / norm2.unsqueeze(1)
    XY = XY / norm2.unsqueeze(1)
    Ax = (Ab * norm2).T.clone()

    rho = 1
    XXinv = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device)*rho)
    U = torch.zeros_like(Ax)
    for itt in range(20):
        Z = (Ax + U) * mask

        U = U + (Ax - Z)

        Ax = XXinv.matmul(XY + rho*(Z-U))

    Ac = Z.T / norm2
    return Ac


def factorize(W, XX, sparsity, nofinal=True, fixmask=None):
    if W.shape[0] == W.shape[1]:
        asp = 0.16
    else:
        asp = 0.25
    W2, Ab, Bb = factorizef(W, XX, asp=asp, sp=1-sparsity, fixmask=fixmask)
    print("err_prefin", (W2 - W).matmul(XX).matmul((W2 - W).T).diag().sum().item())
    if nofinal:
        return W2, Ab.cpu(), Bb.cpu()######nofinal=True


    Ac = finalize(XX, W, Ab, Bb)
    W3 = Ac.matmul(Bb)
    assert W3.shape == W.shape
    print("err_fin   ", (W3 - W).matmul(XX).matmul((W3 - W).T).diag().sum().item())
    print("sparsity check", ((Ac != 0).sum() + (Bb != 0).sum()).item() / W.numel())
    return W3, Ac.cpu(), Bb.cpu()
