import numpy as np
import numpy.linalg as alg
#1 -2 
X=np.array([[2,1],[6,6],[5,3],[3,3],[3,5],[1,2]])
X_pr=X.transpose()
res=X_pr.dot(X)
val_pr,vect_pr=alg.eig(res)
val_pr=sorted(val_pr,key=abs,reverse=True)
print(vect_pr)
#3
vec1=np.array([vect_pr[0][0],vect_pr[1][0]])
norme_vec1=alg.norm(vec1)
u1=vec1*norme_vec1
print(u1)
vec2=np.array([vect_pr[0][1],vect_pr[1][1]])
norme_vec2=alg.norm(vec2)
u2=vec2*norme_vec2
print(u2)
#4
cord_indiv_nvrep=[]
for i in range(6):
    coeff1=np.dot(X[i],u1)
    coeff2=np.dot(X[i],u2)
    cord_indiv_nvrep.append([coeff1,coeff2])
print(cord_indiv_nvrep)
#5
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
#6
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
#7
cord_indiv_nvrep_var=[]
for i in range(2):
    coeff1=np.dot(X_pr[i],v1)
    coeff2=np.dot(X_pr[i],v2)
    cord_indiv_nvrep_var.append([coeff1,coeff2])
print(cord_indiv_nvrep_var)

