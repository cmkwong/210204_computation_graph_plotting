import torch
from torch import nn
from codes import graph

# special size for identity each variable
identity_size = {
    'E': (1,64,10),
    'Q': (5,64,2),
    'F': (2,64,5),
    'H': (2,32,20), 'W': (4,16,20),
    'X': (8,8,64), 'Y': (16,4,64), 'Z': (32,2,64),
    'R': (2,10,64), 'K': (64,1,64)
}

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
        self.H = nn.init.normal_(torch.empty(identity_size['H'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(H_size)
        self.W = nn.init.normal_(torch.empty(identity_size['W'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(W_size)

        self.params = {}
        self.params['F'] = nn.init.normal_(torch.empty(identity_size['F'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(input_size)
        self.params['X'] = nn.init.normal_(torch.empty(identity_size['X'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(X_size)
        self.params['Y'] = nn.init.normal_(torch.empty(identity_size['Y'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(Y_size)
        self.params['Z'] = nn.init.normal_(torch.empty(identity_size['Z'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(Z_size)
        self.params['R'] = nn.init.normal_(torch.empty(identity_size['R'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(R_size)
        self.params['K'] = nn.init.normal_(torch.empty(identity_size['K'], requires_grad=True, dtype=torch.float, device=self.device), mean=0.0, std=0.1).view(K_size)

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
CG = graph.Computation_Graph(identity_size)

fake_sentences = [nn.init.normal_(torch.empty(identity_size['E'], dtype=torch.float, requires_grad=True, device='cuda'), mean=0.0, std=0.1).view(input_size) for _ in range(3)]
fake_question = nn.init.normal_(torch.empty(identity_size['Q'], dtype=torch.float, requires_grad=True, device='cuda'), mean=0.0, std=0.1).view(input_size)

H = model(fake_sentences)
ans = model.answer(fake_question)

CG.recursive_loop(ans.grad_fn, path='')
CG.save(file_name="Testing", view=False)

print()