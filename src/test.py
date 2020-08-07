import torch
from torch import nn, optim
from copy import deepcopy

n = 10000
m = 100

X = torch.rand(n, m)
X_mean = X.mean(dim=0, keepdim=True)
X_std = X.std(dim=0, keepdim=True)
X = (X - X_mean) / X_std
A = (X.t().matmul(X)) / n

v = torch.rand(m, 1)
v = v / torch.norm(v)

criterion = nn.MSELoss()

class Solver(nn.Module):

	def __init__(self, m, lambd=-1, mu=-1):
		super(Solver, self).__init__()
		self.u = nn.Parameter(torch.rand(m, 1))
		self.lambd = lambd
		self.mu = mu
		self.solution = None

	def forward(self, A, v):
		return - (self.u.t().matmul(A).matmul(self.u)[0, 0] + self.lambd * torch.abs(self.u.t().matmul(self.u) - 1) + self.mu * torch.abs(self.u.t().matmul(v)))

def solve(A, v):

	solver = Solver(m, -10, -10)

	optimizer = optim.SGD(solver.parameters(), lr=0.0003, momentum=0.9)

	n_step = 1000

	min_loss = 1e9
	for i in range(n_step):
		optimizer.zero_grad()
		loss = solver(A, v)
		loss.backward()
		if i % 10 == 0:
			print('%d\t%.4f' % (i, loss.item()))
			if loss.item() < min_loss:
				min_loss = loss.item()
				solver.solution = deepcopy(solver.u.data)
		optimizer.step()
	print(min_loss)
	return solver.solution

solution = solve(A, v)
print(solution)
