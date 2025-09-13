import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
# import numpy.matlib
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import cvxopt 
# import cvxopt.cholmod
# import sys
# if sys.platform == 'linux':
#     import torch_sparse_solve 
#-----------------------#
#%%  structural FE
class StructuralFE:
    #-----------------------#
    def initializeSolver(self, data_type, nelx, nely, penal = 3,Emin = 1e-3, Emax = 1.0, nu = 0.3):
        self.Emin = Emin
        self.Emax = Emax
        self.penal = penal
        self.nelx = nelx; # number of elements in x
        self.nely = nely; # numer of elements in y
        self.data_type = data_type
        self.ndof = 2*(nelx+1)*(nely+1) # total DoF
        self.KE=self.getDMatrix_torch(Emax,nu)
        nodenrs = np.arange(0, (1 + nelx) * (1 + nely), dtype=np.int32).reshape((1 + nely, 1 + nelx), order="F")
        self.edofVec = np.reshape(2 * nodenrs[:-1, :-1] + 2, (nelx * nely, 1), order="F")
        self.edofMat = np.tile(self.edofVec, (1, 8)) + np.tile(
            np.concatenate(([0, 1], 2 * nely + np.array([2, 3, 0, 1]), [-2, -1])), (nelx * nely, 1)
        )
        self.iK = np.reshape(np.kron(self.edofMat, np.ones((8, 1), dtype=np.int32)).T, (64 * nelx * nely, 1), order='F').flatten()
        self.jK = np.reshape(np.kron(self.edofMat, np.ones((1, 8), dtype=np.int32)).T, (64 * nelx * nely, 1), order='F').flatten()

        ## PERIODIC BOUNDARY CONDITIONS
        e0 = torch.eye(3, dtype=data_type)
        self.ufixed = torch.zeros((8, 3), dtype=data_type)
        

        self.alldofs = np.arange(0, 2 * (nely + 1) * (nelx + 1), dtype=np.int32)
        n1 = np.concatenate((nodenrs[-1, [0, -1]], nodenrs[0, [-1, 0]]))
        self.d1 = np.reshape(([[(2 * n1)], [2 * n1+1]]), (1, 8), order='F')
        n3 = np.concatenate((nodenrs[1:-1, 0].T, nodenrs[-1, 1:-1]))
        self.d3 = np.reshape(([[(2 * n3)], [2 * n3+1]]), (1, 2 * (nelx + nely - 2)), order='F')
        n4 = np.concatenate((nodenrs[1:-1, -1].flatten(), nodenrs[0, 1:-1].flatten()))
        self.d4 = np.reshape(([[(2 * n4)], [2 * n4+1]]), (1, 2 * (nelx + nely - 2)), order='F')
        self.d2 = np.setdiff1d(self.alldofs, np.hstack([self.d1, self.d3, self.d4]))

        for j in range(3):
            self.ufixed[2:4, j] = torch.tensor([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]], dtype=data_type) @ torch.tensor([nelx, 0], dtype=data_type)
            self.ufixed[6:8, j] = torch.tensor([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]], dtype=data_type) @ torch.tensor([0, nely], dtype=data_type)
            self.ufixed[4:6, j] = self.ufixed[2:4, j] + self.ufixed[6:8, j]

        self.wfixed = torch.cat((
            torch.tile(self.ufixed[2:4, :], (nely - 1, 1)),
            torch.tile(self.ufixed[6:8, :], (nelx - 1, 1))
        ))


    def getDMatrix_torch(self,E,nu):
        k=torch.tensor([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*torch.tensor([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
        return (KE)

    def solve(self, density):
        #density = xPhys.permute(1,0).flatten()
        xPhys = density.view(self.nely, self.nelx).permute(1,0)
        ## INITIALIZE ITERATION
        Q = torch.zeros((3, 3), dtype=self.data_type)

        ## FE-ANALYSIS
        temp1 = torch.unsqueeze(torch.flatten(self.KE),dim=1)
        temp2 = (self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))
        sK=torch.flatten(torch.transpose(temp1*temp2,0,1))
        indices = np.array([self.iK,self.jK])
        ndof = self.alldofs.shape[0]
        K = torch.sparse_coo_tensor(indices, sK, size=(ndof,ndof)).to_dense()

        k1 = K[self.d2][:, self.d2]
        k2 = K[self.d2[:, None], self.d3] + K[self.d2[:, None], self.d4]
        k3 = K[self.d3, self.d2[:, None]] + K[self.d4, self.d2[:, None]]
        k4 = K[self.d3, self.d3.T] + K[self.d4, self.d4.T] + K[self.d3, self.d4.T] + K[self.d4, self.d3.T]

        Kr_top = torch.hstack((k1, k2))
        Kr_bottom = torch.hstack((k3.T, k4))
        Kr = torch.vstack((Kr_top, Kr_bottom))
        rhs = (
            -torch.vstack((K[self.d2[:, None], self.d1], (K[self.d3, self.d1.T] + K[self.d4, self.d1.T]).T)) @ self.ufixed
            -torch.vstack((K[self.d2, self.d4.T].T, (K[self.d3, self.d4.T] + K[self.d4, self.d4.T]).T)) @ self.wfixed
        )

        U = torch.zeros((2 * (self.nely + 1) * (self.nelx + 1), 3), dtype=self.data_type)
        U[self.d1, :] = self.ufixed
        U[np.concatenate((self.d2, self.d3.ravel())), :] = torch.linalg.solve(Kr, rhs)
        # LU, pivots = torch.linalg.lu_factor(Kr)
        # U[np.concatenate((d2, d3.ravel())), :] = torch.linalg.lu_solve(LU, pivots, rhs)
        U[self.d4, :] = U[self.d3, :] + self.wfixed
        for i in range(3):
            for j in range(3):
                U1 = U[:, i]
                U2 = U[:, j]
                temp1 = torch.sum((U1[self.edofMat] @ self.KE) * U2[self.edofMat], dim=1).view(self.nely, self.nelx).permute(1,0)/ (self.nelx * self.nely)
                Q[i, j] = torch.sum((self.Emin + xPhys ** self.penal * (self.Emax - self.Emin)) * temp1)
        return Q

