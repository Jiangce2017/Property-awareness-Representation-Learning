import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def elasticity(xPhys):
    xPhys = xPhys.astype(np.float64)  # ensure input is double precision. Xphys is the matrix

    penal = 1.0 # penalization factor: no effect since it is 1
    nely, nelx = np.shape(xPhys) # since xPhys is a 2x2 then number of elements in x and y are 2

    ## MATERIAL PROPERTIES:
    EO = np.float64(1.0) # Young's Modulus of Solid
    Emin = np.float64(1e-9) #Young's Modulus of a non-solid
    nu = np.float64(0.3) # Poisson's ratio

    ## PREPARE FINITE ELEMENT ANALYSIS builds 8x8 stiffness matrix
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=np.float64) # represents normal strain
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], dtype=np.float64) # cross coupling terms between DOF's
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=np.float64) # scaled by Poisson's ratio (v)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=np.float64) #cross coupling under Poisson's ratio

    #KE builds 2 8x8 matrices and applies them to following formula
    KE = (1 / (1 - nu**2) / 24) * (
        np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]])
    ) # governing equation to build 8x8 global stiffness matrix


    # Creates an array with number of nodes, then reshapes so it is 2d 3x3 array with Fortran order
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1, dtype=np.int32).reshape((1 + nely, 1 + nelx), order="F")

    #Gives top left corner node of each element, and then multiplies by 2 and add 1, then creates 4x1 array
    edofVec = np.reshape(2 * nodenrs[:-1, :-1] + 1, (nelx * nely, 1), order="F")

    #builds 8x1 displacement vector, then repeats each base DOF across 8 columns. Makes 4 copies of the offset array
    #Finally for each element a corresponding 8 degrees of freedom are given
    edofMat = np.tile(edofVec, (1, 8)) + np.tile(
        np.concatenate(([0, 1], 2 * nely + np.array([2, 3, 0, 1]), [-2, -1])), (nelx * nely, 1)
    )

    #Kronecker product # of Elements. Then each element DOF is repeated 8x vertically. Then transposes
    #Reshapes it so 64 entries per element x elements = entries total
    iK = np.reshape(np.kron(edofMat, np.ones((8, 1), dtype=np.int32)).T, (64 * nelx * nely, 1), order='F')

    #each element DOF horizontally and repeated across 8 columns. Each element 64(i,j)
    jK = np.reshape(np.kron(edofMat, np.ones((1, 8), dtype=np.int32)).T, (64 * nelx * nely, 1), order='F')

    ## PERIODIC BOUNDARY CONDITIONS
    e0 = np.eye(3, dtype=np.float64) #Creates a 3x3 identity matrix
    ufixed = np.zeros((8, 3), dtype=np.float64) #Creates array for prescribed displacement at the 8 corner DOF's
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 3), dtype=np.float64) #Created displacement array for all nodes

    alldofs = np.arange(1, 2 * (nely + 1) * (nelx + 1) + 1, dtype=np.int32) #Creates an array of all DOF numbers starting from 1
    #nodenrs is a 3x3 grid of nodes is Fortran order
    n1 = np.concatenate((nodenrs[-1, [0, -1]], nodenrs[0, [-1, 0]])) # gets the 4 corner nodes
    d1 = np.reshape(([[(2 * n1 - 1)], [2 * n1]]), (1, 8), order='F') #converts 4 corner nodes to their DOF numbers (u,v)
    n3 = np.concatenate((nodenrs[1:-1, 0].T, nodenrs[-1, 1:-1])) # gets left and bottom edge nodes
    d3 = np.reshape(([[(2 * n3 - 1)], [2 * n3]]), (1, 2 * (nelx + nely - 2)), order='F') #gets left and bottom edge nodes to their DOF numbers (u,v)
    n4 = np.concatenate((nodenrs[1:-1, -1].flatten(), nodenrs[0, 1:-1].flatten())) # gets right and top edge nodes
    d4 = np.reshape(([[(2 * n4 - 1)], [2 * n4]]), (1, 2 * (nelx + nely - 2)), order='F') # converts right/top edge nodes to their DOF numbers

    #combines all boundary DOF into a single array then sorts by ascending number and finds DOF's not used
    d2 = np.setdiff1d(alldofs, np.hstack([d1, d3, d4]))

    for j in range(3): #runs 3 times for unit strain 11, 22 and shear 12

        #computes displacement at bottom-right corner due to strain field. Uses u=strain*x
        ufixed[2:4, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]], dtype=np.float64) @ np.array([nelx, 0], dtype=np.float64)

        #At top left corner, gets the displacement due to strain
        ufixed[6:8, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]], dtype=np.float64) @ np.array([0, nely], dtype=np.float64)

        #displacement at top-right corner: vector sum of bottom-right and top=left displacement
        ufixed[4:6, j] = ufixed[2:4, j] + ufixed[6:8, j]

    #Takes bottom right corner displacement for 3 strain directions and repeats it nely-1 vertically
    #Takes top left corner nodes displacement repeats it nelx-1 times vertically
    #Verically stacks the two blocks into one big matrix
    wfixed = np.concatenate((
        np.tile(ufixed[2:4, :], (nely - 1, 1)),
        np.tile(ufixed[6:8, :], (nelx - 1, 1))
    ))

    ## INITIALIZE ITERATION
    qe = np.empty((3, 3), dtype=object) #Prepares 3x3 array to store element strain energy density
    Q = np.zeros((3, 3), dtype=np.float64) # Initializes 3x3 matrix which stores homogenized elasticity tensor

    ## FE-ANALYSIS
    #Vectorize domain. It scales K by young's modulus and penal
    sK = (KE.flatten(order='F')[:, np.newaxis] *
         (Emin + xPhys.flatten(order='F').T ** penal * (EO - Emin)))[np.newaxis, :].reshape(-1, 1, order='F')

    #assembles global stiffness matrix K from elements
    K = sp.coo_matrix((sK.flatten(order='F'), (iK.flatten(order='F') - 1, jK.flatten(order='F') - 1)),
                      shape=(alldofs.shape[0], alldofs.shape[0]))
    K = (K + K.transpose()) / 2 # Ensures that K is symmetric
    K = sp.csr_matrix(np.nan_to_num(K.toarray(), nan=0.0)) #Replaces any NaN with 0
    K = K.tocsc() #converts matrix to compressed sparse column K*U=F

    k1 = K[d2 - 1][:, d2 - 1] #square matrix with internal-internal DOF's (top-left)
    k2 = K[d2[:, None] - 1, d3 - 1] + K[d2[:, None] - 1, d4 - 1] #Extracts the internal-boundary DOF (top-right)
    k3 = K[d3 - 1, d2[:, None] - 1] + K[d4 - 1, d2[:, None] - 1] #Extracts the boundary-internal DOF (bottom-left) Transpose of k2
    k4 = K[d3 - 1, d3.T - 1] + K[d4 - 1, d4.T - 1] + K[d3 - 1, d4.T - 1] + K[d4 - 1, d3.T - 1] #Extracts boundary-boundary DOF (bottom-right)

    Kr_top = sp.hstack((k1, k2)) #horizontally stacks k1 and k2
    Kr_bottom = sp.hstack((k3.T, k4)) #horizontally stacks k3.T and k4 into bottom row of Kr
    Kr = sp.vstack((Kr_top, Kr_bottom)).tocsc() #builds the reduced stiffness matrix

    #Builds right-hand side vector
    #Following code represents the effect of known displacements at the corners nodes d1 on d2
    #Second line represents the effect of periodic offset displacements on internal + boundary DOF's
    rhs = (
        -sp.vstack((K[d2[:, None] - 1, d1 - 1],
                    (K[d3 - 1, d1.T - 1] + K[d4 - 1, d1.T - 1]).T)) @ ufixed
        - sp.vstack((K[d2 - 1, d4.T - 1].T,
                     (K[d3 - 1, d4.T - 1] + K[d4 - 1, d4.T - 1]).T)) @ wfixed
    )

    U[d1 - 1, :] = ufixed #sets the corner nodes to their known fixed displacements
    U[np.concatenate((d2 - 1, d3.ravel() - 1)), :] = spsolve(Kr, rhs) #Solves the reduced sytem Kr*Q=rhs (vector)
    U[d4 - 1, :] = U[d3 - 1, :] + wfixed

    for i in range(3): #loops over all pairs of components of strain tensor
        for j in range(3): #does for i and j which represent strain loading directions
            U1 = U[:, i] #displacement vector from unit strain i
            U2 = U[:, j] #displacement vector from unit strain j

            #Computes the average strain energy density for strain components i and j over all elements
            qe[i, j] = np.reshape(np.sum((U1[edofMat - 1] @ KE) * U2[edofMat - 1], axis=1), (nely, nelx), order='F') / (nelx * nely)

            #Integrates the energy density across the material domain to get the effective elasticity tensor
            Q[i, j] = np.sum((Emin + xPhys ** penal * (EO - Emin)) * qe[i, j])

    Q[np.abs(Q) < 1e-6] = 0  # if value close to 0, it will be 0
    return Q
