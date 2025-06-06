import torch
import torch.nn as nn


def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()))
        """if w == 0:
            self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)),requires_grad=True)
        else:
            self.w = w
        if w == 0:
            self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)),requires_grad=True)
        else:
            self.v = v
        if w == 0:
            self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()),requires_grad=True)
        else:
            self.theta = theta"""

    def forward(self, x):
        exp_i = torch.zeros((4,4))
        w_skewsym = vec2ss_matrix(self.w)
        v_skewsym = vec2ss_matrix(self.v)
        exp_i[:3, :3] = torch.eye(3) + torch.sin(self.theta) * w_skewsym + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3) * self.theta + (1 - torch.cos(self.theta)) * w_skewsym + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.
        # import pdb;pdb.set_trace()
        T_i = torch.matmul(exp_i, x)
        return T_i

def camera_trans(w,v,c,pose):
    exp_i = torch.zeros((4, 4))
    w_skewsym = vec2ss_matrix(w)
    v_skewsym = vec2ss_matrix(v)
    exp_i[:3, :3] = torch.eye(3) + torch.sin(c) * w_skewsym + (1 - torch.cos(c)) * torch.matmul(
        w_skewsym, w_skewsym)
    exp_i[:3, 3] = torch.matmul(torch.eye(3) * c + (1 - torch.cos(c)) * w_skewsym + (
                c - torch.sin(c)) * torch.matmul(w_skewsym, w_skewsym), v)
    exp_i[3, 3] = 1.
    new_pose = torch.matmul(exp_i, pose)
    return new_pose