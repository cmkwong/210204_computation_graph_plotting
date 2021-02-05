import torch
from torch import nn
from codes import  graph

# x = nn.init.normal_(torch.empty((4,1), requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)
# y = nn.init.normal_(torch.empty((4,1), requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)
# w = nn.init.normal_(torch.empty((1,4), requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)
# z = nn.init.normal_(torch.empty((1,1), requires_grad=True, dtype=torch.float), mean=0.0, std=0.1)
x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
w = torch.Tensor([1, 2, 3, 4]).requires_grad_()
z = torch.Tensor([1, 1, 1, 1]).requires_grad_()

for _ in range(3):
    k = x + y
    z = z * k
z.retain_grad()

o = w.matmul(z)
o.retain_grad()
o.backward()

print('x.requires_grad:', x.requires_grad) # True
print('y.requires_grad:', y.requires_grad) # True
print('z.requires_grad:', z.requires_grad) # True
print('w.requires_grad:', w.requires_grad) # True
print('o.requires_grad:', o.requires_grad) # True

print('x.grad:', x.grad) # tensor([1., 2., 3., 4.])
print('y.grad:', y.grad) # tensor([1., 2., 3., 4.])
print('w.grad:', w.grad) # tensor([ 4.,  6.,  8., 10.])
print('z.grad:', z.grad) # None
print('o.grad:', o.grad) # None

# plot graph
CG = graph.Computation_Graph()
CG.recursive_loop(o.grad_fn)
CG.save(file_name="Testing", view=True)
print()