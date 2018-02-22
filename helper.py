import numpy as np
import math

def gaussian_multiD(x, A, mu):
	num_D=len(x[0])
	return np.sqrt(np.linalg.det(A)/pow(2*np.pi, num_D))*np.exp(-1/2*(x-mu)*A*(x-mu).T)