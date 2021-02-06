import torch
from torch import nn
from codes import graph

input_size = (64,10)
H_size, W_size = (64,20), (64,20)
X_size, Y_size, Z_size = (64,64), (64,64), (64,64)
R_size, K_size = (20,64), (64,64)

def unitVector_2d(tensors, dim=0):
    """
    :param tensors: torch.Tensor(size=(n,m))
    :return: return normalized torch.Tensor(size=(n,m))
    """
    magnitude = tensors.detach().pow(2).sum(dim=dim).sqrt().unsqueeze(dim)
    unit_tensors = tensors / magnitude
    return unit_tensors

class Test_entNet_Model(nn.Module):

    def __init__(self):
        super(Test_entNet_Model, self).__init__()
        self.device = "cuda"

        # embedding parameters
        self.H = nn.init.normal_(torch.empty(H_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)
        self.W = nn.init.normal_(torch.empty(W_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)

        self.params = nn.ParameterDict({
            'F': nn.Parameter(nn.init.normal_(torch.empty(input_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # shared parameters
            'X': nn.Parameter(nn.init.normal_(torch.empty(X_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Y': nn.Parameter(nn.init.normal_(torch.empty(Y_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'Z': nn.Parameter(nn.init.normal_(torch.empty(Z_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),

            # answer parameters
            'R': nn.Parameter(nn.init.normal_(torch.empty(R_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1)),
            'K': nn.Parameter(nn.init.normal_(torch.empty(K_size, requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1))
        })

        # dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, Es):
        for E in Es:
            self.s = torch.mul(self.params['F'], E).sum(dim=1).unsqueeze(1)  # (64*1)
            self.G = nn.Softmax(dim=1)((torch.mm(self.s.t(), self.H) + torch.mm(self.s.t(), self.W)))  # (1*m)
            self.new_H = nn.Sigmoid()(torch.mm(self.dropout(self.params['X']), self.H) +
                                      torch.mm(self.dropout(self.params['Y']), self.W) +
                                      torch.mm(self.dropout(self.params['Z']), self.s))  # (64*m)
            self.H = unitVector_2d(self.H + torch.mul(self.G, self.new_H), dim=0)  # (64*m)

        return self.H

    def answer(self, Q):
        Q.requires_grad_()
        self.q = torch.mul(self.params['F'], Q).sum(dim=1).unsqueeze(1)  # (64*1)
        self.p = nn.Softmax(dim=1)(torch.mm(self.q.t(), self.H))  # (1*m)
        self.u = torch.mul(self.p, self.H).sum(dim=1).unsqueeze(1)  # (64*1)
        # self.unit_params('R', dim=1)
        self.ans_vector = torch.mm(self.params['R'], nn.Sigmoid()(self.q + torch.mm(self.params['K'], self.u)))  # (k,1)
        self.ans = nn.LogSoftmax(dim=1)(self.ans_vector.t())
        return self.ans

#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#

model = Test_entNet_Model()
CG = graph.Computation_Graph()

fake_sentences = [nn.init.normal_(torch.empty(input_size, dtype=torch.float, device='cuda'), mean=0.0, std=0.1) for _ in range(3)]
fake_question = nn.init.normal_(torch.empty(input_size, dtype=torch.float, device='cuda'), mean=0.0, std=0.1)

H = model(fake_sentences)
ans = model.answer(fake_question)

CG.recursive_loop(H.grad_fn)
CG.save(file_name="Testing", view=True)

print()