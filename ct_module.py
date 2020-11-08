import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedSoftPlus(torch.nn.Module):
    def __init__(self, init_beta=1.0, threshold=20):
        super().__init__()
        # keep beta > 0
        self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log())
        self.threshold = 20
    def forward(self, x):
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(beta_x < 20, torch.log1p(beta_x.exp()) / beta, x)

sigma = LearnedSoftPlus()

class DGMCell(nn.Module):
    def __init__(self, d, M, growing, weight_norm):
        super().__init__()
        wn = WN if weight_norm else lambda x: x

        self.Uz = wn(nn.Linear(d, M, bias=False))
        self.Ug = wn(nn.Linear(d, M, bias=False))
        self.Ur = wn(nn.Linear(d, M, bias=False))
        self.Uh = wn(nn.Linear(d, M, bias=False))

        self.Wz = wn(nn.Linear(M, M))
        self.Wg = wn(nn.Linear(M, M))
        self.Wr = wn(nn.Linear(M, M))
        self.Wh = wn(nn.Linear(M, M))

        self.A = (lambda x: x) if growing else sigma

    def forward(self, SX):
        S, X = SX
        Z = sigma(self.Uz(X) + self.Wz(S))
        G = sigma(self.Ug(X) + self.Wg(S))
        R = sigma(self.Ur(X) + self.Wr(S))
        H = self.A(self.Uh(X) + self.Wh(S*R))
        S = (1-G)*H + Z*S

        return S, X

def _set_convert(flag):
    if flag: return lambda X: X[0]
    return lambda X: torch.stack(X, -1)

class RNNLikeDGM(DGMCell):
    """
    Args:
    -----
    d_in and d_out- input and ouput dimensions of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(
            self, d_in, d_out, M=50, L=3,
            growing=False, as_array=True, weight_norm=False):
        super().__init__(d_in, M, growing, weight_norm)
        self.L = L

        wn = WN if weight_norm else lambda x: x
        self.W0 = wn(nn.Linear(d_in, M))
        self.W1 = wn(nn.Linear(M, d_out))
        self._convert = _set_convert(as_array)
        print('hello')

    def forward(self, *X):
        X = self._convert(X)
        S = sigma(self.W0(X))
        for l in range(self.L):
            Z = sigma(self.Uz(X) + self.Wz(S))
            G = sigma(self.Ug(X) + self.Wg(S))
            R = sigma(self.Ur(X) + self.Wr(S))
            H = self.A(self.Uh(X) + self.Wh(S*R))
            S = (1-G)*H + Z*S
        return self.W1(S)

class RNNEncoderLikeDGM(DGMCell):
    """
    Args:
    -----
    d_in and d_out- input and ouput dimensions of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(
            self, d_in, M=50, L=3,
            growing=False, as_array=True, weight_norm=False):
        super().__init__(d_in, M, growing, weight_norm)
        self.L = L

        wn = WN if weight_norm else lambda x: x
        self.W0 = wn(nn.Linear(d_in, M))
        self.W1 = wn(nn.Linear(M, M))
        
        self._convert = _set_convert(as_array)

    def forward(self, *X):
        X = self._convert(X)
        S = sigma(self.W0(X))
        for l in range(self.L):
            Z = sigma(self.Uz(X) + self.Wz(S))
            G = sigma(self.Ug(X) + self.Wg(S))
            R = sigma(self.Ur(X) + self.Wr(S))
            H = self.A(self.Uh(X) + self.Wh(S*R))
            S = (1-G)*H + Z*S
        return self.W1(S)
        
    def get_last_shared_layer(self):
        return self.W1
    
class Decoder(nn.Module):
    def __init__(self, M, d_out, weight_norm=False):
        super().__init__()
        wn = WN if weight_norm else lambda x: x
        self.linear = wn(nn.Linear(M, d_out))
    
    def forward(self, feature):
        return self.linear(feature).squeeze_(-1)

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.f = nn.Sequential(nn.Linear(2, 50), 
                          nn.Tanh(), 
                          nn.Linear(50, 50),
                          nn.Tanh(), 
                          nn.Linear(50, 50),
                          nn.Tanh(), 
                          nn.Linear(50, 50),
                          nn.Tanh())

    # x represents our data
    def forward(self, feature):
        return self.f(feature)

class HDL(nn.Module):
    def __init__(self):
        super(HDL, self).__init__()
        self.fc = nn.Linear(50, 1)
        
    def forward(self, feature):
        return self.fc(feature).squeeze_(-1)
    
class FFN_mtl(nn.Module):
    def __init__(self):
        super(FFN_mtl, self).__init__()
        self.f = nn.Sequential(nn.Linear(2, 50), 
                          nn.Tanh(), 
                          nn.Linear(50, 50),
                          nn.Tanh(), 
                          nn.Linear(50, 50),
                          nn.Tanh())
        self.W1 = nn.Sequential(nn.Linear(50, 50), nn.Tanh())
        self.fc1 = nn.Linear(50, 1)
        self.fc2 = nn.Linear(50, 1)

    # x represents our data
    def forward(self, feature):
        enc = self.f(feature)
        enc = self.W1(enc)
        return self.fc1(enc).squeeze_(-1), self.fc2(enc).squeeze_(-1)
    
# class RNNLikeDGM(DGMCell):
#     """
#     Args:
#     -----
#     d_in and d_out- input and ouput dimensions of the problem
#     M - layers' width
#     L - recurrency depth
#     """
#     def __init__(
#             self, d_in, d_out, M=50, L=3,
#             growing=False, as_array=True, weight_norm=False):
#         super().__init__(d_in, M, growing, weight_norm)
#         self.L = L

#         wn = WN if weight_norm else lambda x: x
#         self.W0 = wn(nn.Linear(d_in, M))
#         self.W1 = wn(nn.Linear(M, M))
        
#         self.out1 = wn(nn.Linear(M, d_out))
#         self.out2 = wn(nn.Linear(M, d_out))
        
#         self._convert = _set_convert(as_array)

#     def forward(self, *X):
#         X = self._convert(X)
#         S = sigma(self.W0(X))
#         for l in range(self.L):
#             Z = sigma(self.Uz(X) + self.Wz(S))
#             G = sigma(self.Ug(X) + self.Wg(S))
#             R = sigma(self.Ur(X) + self.Wr(S))
#             H = self.A(self.Uh(X) + self.Wh(S*R))
#             S = (1-G)*H + Z*S
#         S = self.W1(S)
        
#         return self.out1(S).squeeze_(-1), self.out2(S).squeeze_(-1)
        
    def get_last_shared_layer(self):
        return self.W1
    
def get_model(d_in, d_out, M):
    model = {}
    model['rep'] = RNNEncoderLikeDGM(d_in, M=M)
    model['f'] = Decoder(M, d_out)
    model['s'] = Decoder(M, d_out)
    return model

def get_model_FFN(d_in, d_out, M):
    model = {}
    model['rep'] = FFN()
    model['f'] = HDL()
    model['s'] = HDL()
    return model

class CrossStich(nn.Module):
    def __init__(self,):
        super(CrossStich, self).__init__()
        self.transform = nn.Parameter(data=torch.eye(2), requires_grad=True)
    def forward(self, input_1, input_2):
        return self.transform[0][0]*input_1 + self.transform[0][1]*input_2, self.transform[1][0]*input_1 + self.transform[1][1]*input_2
    
class LargeCrossStich(nn.Module):
    def __init__(self, size):
        super(LargeCrossStich, self).__init__()
        self.transform = nn.Parameter(data=torch.eye(2*size), requires_grad=True)
    def forward(self, input_1, input_2):
        out = torch.mm(torch.cat([input_1, input_2], 1), self.transform)
        return out[:, 0:input_1.shape[1]], out[:, input_1.shape[1]:]