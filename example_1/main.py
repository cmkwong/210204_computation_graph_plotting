import torch
from torch import nn
from codes import graph

class Test_Model(nn.Module):

    def __init__(self):

        super(Test_Model, self).__init__()
        self.device = "cuda"
        self.a = nn.init.normal_(torch.empty((10,1), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)
        self.b = nn.init.normal_(torch.empty((10,10), requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)

    def forward(self):

        # for i in range(2):
        #     foo = "c" + str(i)
        #     exec(foo + " = self.a + self.b")
        c = self.a + self.b
        d = c * self.b
        e = d / self.a
        f = self.a * e
        f = self.b * f

        return f

model = Test_Model()
value = model.forward()

CG = graph.Computation_Graph()
CG.recursive_loop(value.grad_fn)
CG.save(file_name="Testing", view=True)

print()

