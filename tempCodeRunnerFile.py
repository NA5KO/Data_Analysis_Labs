import numpy as np
import numpy.linalg as alg
#1 les poids relatifs 
X_inti=np.array([[5,3,2],[2,6,1],[4,1,7]])
total_weight=0
for i in range(3):
    total_weight+=X_inti[i][2]
print(total_weight)
x1_RW = X_inti[0][2]/total_weight
x2_RW = X_inti[1][2]/total_weight
x3_RW = X_inti[2][2]/total_weight
print(x1_RW)
print(x2_RW)
print(x3_RW)

#2 la matrice a considerer est 
X=np.array([[5,3],[2,6],[4,1]])
print(X)

# la matrice D 1/2
D_demi = np.diag(np.sqrt([x1_RW,x2_RW,x3_RW]))
print(D_demi)

#3 la matrice de travail est D1/2 * X
X_prime = np.dot(D_demi,X)
print("la matrice de travail est \n",X_prime)

#la matrice a diagonlaiser est X_prime * X_prime_transpose
X_prime_transpose = X_prime.transpose()
res=np.dot(X_prime_transpose,X_prime)
print("la matrice a diagomaliser est \n",res)

#4
val_pr,vect_pr=alg.eig(res)
print(val_pr)
print(vect_pr)