import numpy as np
from copy import deepcopy

def minres(A, b, M_inv, n):
  shift = 0.
  x = np.zeros(n)
  y = deepcopy(b)
  y = M_inv @ y
  w = np.zeros(n)
  w2 = np.zeros(n)
  r1 = deepcopy(b)
  r2 = deepcopy(r1)

  beta1 = np.sqrt(b.T @ y)
  
  if beta1 < 1e-6:
    return x
  
  oldb, beta, dbar, epsln  = 0., beta1, 0., 0.
  phibar = beta1
  cs, sn = -1, 0
  
  for iter in range(n):
    s = 1. / beta
    v = s * y
    y = A @ v - shift * v
    if iter > 0: y = y - (beta / oldb) * r1
    alfa = v.T @ y
    y = (-alfa / beta) * r2 + y
    r1 = deepcopy(r2)
    r2 = deepcopy(y)
    y = M_inv @ r2
    oldb = beta
    beta = np.sqrt(r2.T @ y)
    
    oldeps = epsln
    delta = cs * dbar + sn * alfa
    gbar = sn * dbar - cs * alfa
    epsln = sn * beta
    dbar = -cs * beta
    
    gamma = np.sqrt(gbar**2 + beta**2)
    gamma = max(gamma, 1e-15)
    cs = gbar / gamma
    sn = beta / gamma
    phi = cs * phibar
    phibar = sn * phibar
    
    denom = 1. / gamma
    w1 = deepcopy(w2)
    w2 = deepcopy(w)
    w = (v - oldeps * w1 - delta * w2) * denom
    x += phi * w
    if abs(phibar) / beta1 < 1e-6:
      break
  
  return x
  
def generate_indefinite_symmetric_matrix(n):
  A = np.random.randn(n, n)
  Q, R = np.linalg.qr(A)
  D = np.diag(np.random.choice([-1, 10], size=n))
  M = Q.dot(D).dot(Q.T)
  return M

# A = generate_indefinite_symmetric_matrix(16)
A = np.loadtxt("A00.txt")
print(f"A cond: {np.linalg.cond(A)}")
b = np.random.randn(16)
# b = np.zeros(16)

x = np.linalg.solve(A, b)
print(f"direct res: {np.linalg.norm(A @ x - b)}")

M1 = np.eye(16)
x1 = minres(A, b, M1, 16)
print(f"no precond res: {np.linalg.norm(A @ x1 - b)}")

M2_inv = np.diag(np.abs(1. / A.diagonal()))
print(f"diag precond cond: {np.linalg.cond(M2_inv @ A)}")
x2 = minres(A, b, M2_inv, 16)
print(f"diag precond res: {np.linalg.norm(A @ x2 - b)}")

print(f"no precond err: {np.linalg.norm(x1 - x)}")
print(f"diag precond err: {np.linalg.norm(x2 - x)}")