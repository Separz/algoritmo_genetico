import random as ra
import numpy as np
import matplotlib.pyplot as plt

#########################################################################
#FUNCION CUANTIL PARA SELECCIONAR AL PADRE DE ACUERDO A LA PROBABILIDAD
#ACUMULADA
#########################################################################

def cuantile(prob_acumulada, poblacion):
  '''selecciona un cromosma ganador de acuerdo a la probabilidad acumulada de cada uno'''
  p = ra.uniform(0, 1) #aleatorio decimal entre 0 y 1
  for i, prob in enumerate(prob_acumulada):
    if p < prob: # si es mayor o igual a la probabilidad acumulada 
      print(f'Cuantil Seleccionado: {p}\nProbabilidad: {prob}')
      print(f'Cromosoma seleccionado: {poblacion[i]} en Posicion {i}') # obtiene el valor
      break

def mutar(poblacion, p_mut):
  #obtener punto de corte
  n_crom, n_votos = poblacion.shape # 2, 12
  #punto de corte randomico entre 0 y n_votos-1 (posiciones)
  pos = ra.randrange(n_votos)

  for i in range(n_crom):
    orig = poblacion[i][pos]
    mutado = poblacion[i][pos] ^ 1 #aplicar xor
    print(f"la posicion {pos} del cromosoma {i} muto de {orig} a {mutado}!" )

  ...





#def cruzar(poblacion, p_mut):
#  ...


pobl_test = np.array([[1,0,1,0,1,1,1,0,0,1,1,0],
                      [0,0,1,0,1,0,1,1,0,1,1,1]], dtype=np.uint8)


p_mut = 0.5




mutar(pobl_test, p_mut)



#poblacion de cromosomas (individuos)
poblacion = [[0,0,0],[1,0,1],[0,1,0],[1,1,1],[1,1,0],[1,1,0],[1,1,0],[1,0,0],[1,1,1],[0,1,1],[0,0,1]]

prob_acumuladas = [0.6000251668795754, 0.8400352336314056, 0.9360392603321377, 0.9744408710124305, 
                   0.9898015152845476, 0.9959457729933945, 0.9984034760769332, 0.9993865573103488, 
                   0.999779789803715, 0.9999370828010614, 1.0]


cuantile(prob_acumuladas, poblacion)
