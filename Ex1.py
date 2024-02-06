'''
1-pour commencer on a X'X est une matrice carree,en plus ccette matrice est symetrique a coefficients reels d'ou
 X'X est diagnolisable
2-
dans ce question on definit notre matrice X puis on fait le produit avec son transpose d'ou on obtient X'X
puis a laide de la fonction eig on prend notre vecteur propre associe a chaque valeur propre

import numpy as np
import numpy.linalg as alg

X=np.array([[2,1],[6,6],[5,3],[3,3],[3,5],[1,2]])
X_pr=X.transpose()
res=X_pr.dot(X)
val_pr,vect_pr=alg.eig(res)
val_pr=sorted(val_pr,key=abs,reverse=True)
print(vect_pr)
3-
dans ce question on forme une base orthornorme a laide des vecteurs propres ,en fait, on les rend unitaire puisque il sont deja orthogonaux

vec1=np.array([vect_pr[0][0],vect_pr[1][0]])
norme_vec1=alg.norm(vec1)
u1=vec1*norme_vec1
print(u1)
vec2=np.array([vect_pr[0][1],vect_pr[1][1]])
norme_vec2=alg.norm(vec2)
u2=vec2*norme_vec2
print(u2)
4-
dans ce question on cherche les coefficients des individus dans notre nouvelle base (obtenue a laide des vecteurs propres)
en utlisant le produit scalaire

cord_indiv_nvrep=[]
for i in range(6):
    coeff1=np.dot(X[i],u1)
    coeff2=np.dot(X[i],u2)
    cord_indiv_nvrep.append([coeff1,coeff2])
print(cord_indiv_nvrep)
5-
a l'aide de la bib matplotlib et de la pyplot on essaye de representer la base canonique et les individus dans cette base et on meme temps on visualise la nouvelle base et la nouvelle dispersion des point par rapports au deux vecteurs u1 et u2

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], label='Repère Canonique', marker='o')
plt.scatter(np.array(cord_indiv_nvrep)[:, 0], np.array(cord_indiv_nvrep)[:, 1], label='Nouveau Repère', marker='x')

plt.quiver(0, 0, u1[0], u1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vecteur propre u1')
plt.quiver(0, 0, u2[0], u2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vecteur propre u2')

x_vals_d1 = np.linspace(-max(cord_indiv_nvrep)[0], max(cord_indiv_nvrep)[0], 2)
y_vals_d1 = (u1[1]/u1[0]) * x_vals_d1
plt.plot(x_vals_d1, y_vals_d1, '--', color='r', label='Droite d1')

x_vals_d2 = np.linspace(-max(cord_indiv_nvrep)[0], max(cord_indiv_nvrep)[0], 2)
y_vals_d2 = (u2[1]/u2[0]) * x_vals_d2
plt.plot(x_vals_d2, y_vals_d2, '--', color='b', label='Droite d2')

for i, txt in enumerate(X):
    plt.annotate(txt, (X[i, 0], X[i, 1]))

plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.legend()
plt.grid(True)
plt.show()
6-
dans cette question ,on travaille maintetnat sur les nuages des variables(N(j)) cad les colonnes de X ainsi meme travail que les question precedents

res_var=X.dot(X_pr)
val_pr_var,vect_pr_var=alg.eig(res_var)
val_pr_var=sorted(val_pr_var,key=abs,reverse=True)

vec1_var=vect_pr_var[:,0]
vec2_var=vect_pr_var[:,2]

norme_vec1_var=alg.norm(vec1_var)
v1=vec1_var*norme_vec1_var
norme_vec2_var=alg.norm(vec2_var)
v2=vec2_var*norme_vec2_var
print(v1,v2)
7-
meme travail mais cette fois sur les variables(colonnes)
cord_indiv_nvrep_var=[]
for i in range(2):
    coeff1=np.dot(X_pr[i],v1)
    coeff2=np.dot(X_pr[i],v2)
    cord_indiv_nvrep_var.append([coeff1,coeff2])
print(cord_indiv_nvrep_var)
8-
dans cette question , on a ajoute une nouvelle ligne (individu)et une nouvelle colonne (variable) et on les ecrits dans leurs base respectives

vect_illust=np.array([3,9])
print(vect_illust)
coeff_vect_illust=[np.dot(vect_illust,u1),np.dot(vect_illust,u2)]
print(coeff_vect_illust)
vect_supp=np.transpose(np.array([1,2,1,1,3,2]))
print(vect_supp)
coeff_vect_supp=[np.dot(vect_supp,v1),np.dot(vect_supp,v2)]
print(coeff_vect_supp)

'''