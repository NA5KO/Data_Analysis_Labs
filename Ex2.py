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
print("la matrice diagonale D1/2 est \n",D_demi,"\n")

#3 la matrice de travail est D1/2 * X
X_prime = np.dot(D_demi,X)
print("la matrice de travail est \n",X_prime,"\n")

#la matrice a diagonlaiser est X_prime * X_prime_transpose
X_prime_transpose = X_prime.transpose()
res=np.dot(X_prime_transpose,X_prime)
print("la matrice a diagomaliser est \n",res)

#4
val_pr,vect_pr=alg.eig(res)
print("les valeurs propres sont \n",val_pr,"\n")
print("les vecteurs propores associes sont\n",vect_pr,"\n")

#5
val_pr=sorted(val_pr,key=abs,reverse=True)
print(val_pr)
vec1=np.array([vect_pr[0][0],vect_pr[1][0]])
norme_vec1=alg.norm(vec1)
u1=vec1*norme_vec1
print("le vecteur u1 est ",u1,"\n")
vec2=np.array([vect_pr[0][1],vect_pr[1][1]])
norme_vec2=alg.norm(vec2)
u2=vec2*norme_vec2
print("le vecteur u2 est ",u2,"\n")

#6
cord_indiv_nvrep=[]
for i in range(3):
    coeff1=np.dot(X[i],u1)
    coeff2=np.dot(X[i],u2)
    cord_indiv_nvrep.append([coeff1,coeff2])
print("les nouveaux cordonnes sont ",cord_indiv_nvrep)










