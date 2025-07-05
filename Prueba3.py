import json
import math
import numpy as np
import matplotlib.pyplot as plt
import random as ra
from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix
import time

# generador de numeros aleatorios
rng = np.random.default_rng()

def getDatos(path):
  '''obtener los datos del json'''
  # cargar el archivo JSON
  with open(path) as f:
    data = json.load(f)

  points = []      # lista de puntos [x, y]

  # extraer datos relevantes
  for rollcall in data['rollcalls']:
    for vote in rollcall['votes']:
      points.append([vote['x'], vote['y']])  # guardar [x,y] 

  n_votes = len(points)
  return np.array(points), n_votes  # convertir a array np


def generarPoblacionInicial(n_pob, n_votes, quorum):
  poblacion = np.empty((n_pob, n_votes), dtype=np.uint8) # crear vacio
  for i in range(n_pob):
    poblacion[i] = generarCromosoma(n_votes, quorum)
  return poblacion


def generarCromosoma(total, quorum):
  '''genera un cromosoma inicial que cumple con la coalicion minima ganadora'''
  # vector ceros
  cromosoma = np.zeros(total, dtype=np.uint8)
  # seleccionar posiciones aleatorias para los unos
  posicion = np.random.choice(total, size=quorum, replace=False)
  # asignar unos a las posiciones
  cromosoma[posicion] = 1

  return cromosoma


def graf_polig_convexo(cromosoma, pts, ax):
  '''crea el grafico interactivo para mostrar los puntos en cada generacion'''
  # filtrar puntos
  pts_select = pts[cromosoma == 1]
  pts_no_select = pts[cromosoma == 0]

  # graficar puntos
  ax.scatter(pts_no_select[:, 0], pts_no_select[:, 1], color='blue', label='Puntos no seleccionados')
  ax.scatter(pts_select[:, 0], pts_select[:, 1], color='red', label='Puntos seleccionados')
  
  # calcular y graficar el poligono convexo
  if len(pts_select) >= 3:
    hull = ConvexHull(pts_select)
    # dibujar aristas
    for simplex in hull.simplices:
      ax.plot(pts_select[simplex, 0], pts_select[simplex, 1], 'm-')
    # dibujar vertices
    ax.plot(pts_select[hull.vertices, 0], pts_select[hull.vertices, 1], 'o', mec='red', color='none', lw=1, markersize=8)
  else:
    print("No hay puntos suficientes para formar el poligono convexo")
  
  ax.set_xlabel('Posición Política')
  ax.set_ylabel('Posición Política (otro)')
  ax.set_title('Algoritmo Genético para MWC')
  ax.grid(True, linestyle='--', linewidth=0.5)
  ax.axis([-1.1, 1.1, -1.1, 1.1])
  ax.legend()
  plt.show()


def calcularFitness(cromosoma, matriz_distancias):
  '''crear una matriz de distancias de todos los puntos del cromosoma 
  y calcula el fitness del mismo'''
  # crear la matriz de distancias con los puntos seleccionados
  submatriz = matriz_distancias[cromosoma == 1][:, cromosoma == 1] #donde hayan votos
  # calcular suma de distancias unicas sin diagonal y sin duplicados (triangular superior)
  triang_superior = np.triu(submatriz, k=1)
  fitness = np.sum(triang_superior) #obtener distancia total
  return fitness


def ordenarPoblacion(poblacion):
  '''Se selecciona al mejor individuo de acuerdo a la probabilidad del fitness'''

  global mejor_dist_anterior
  global mejor_dist_actual

  # calular fitness de cada cromosoma
  fitness_valores = np.array([calcularFitness(cromosoma, matriz_distancias) for cromosoma in poblacion])

  # ordenar de menor a mayor fitness
  indices_ordenados = np.argsort(fitness_valores)
  poblacion_ordenada = poblacion[indices_ordenados]
  # obtener la mejor distrancia actual
  mejor_dist_actual = fitness_valores[0]
  # verificar si hay mejora
  if mejor_dist_actual < mejor_dist_anterior:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Generacion Actual: {cont_gen}')
    print(f'Z = Mejor distancia total actual: {mejor_dist_actual}')
    print(f"Tiempo de ejecucion: {elapsed_time:.4f} segundos\n")
    mejor_dist_anterior = mejor_dist_actual
  return poblacion_ordenada


def prob_posicion(n_crom, prob):
  '''calcular la probabilidad de elegir al individuo segun la posicion 
  en el indice. Retorna un arreglo de probabilidades'''
  indices = np.arange(1, n_crom+1)
  probabilidades = prob * (1-prob)**(indices-1)
  return probabilidades / np.sum(probabilidades)


def crossover(padre1, padre2, p_corte):
  '''Cruzar dos pares de cromosomas dado un punto de corte
  Devuelve dos cromosomas hijos'''

  hijo1 = np.concatenate((padre1[:p_corte], padre2[p_corte:]))
  hijo2 = np.concatenate((padre2[:p_corte], padre1[p_corte:]))

  return hijo1, hijo2


def mutacion(crom_original, p_mut):
  '''Cambiar un dato aleatorio de un cromosoma mediante la probabilidad p_mut'''
  crom_mutado = crom_original.copy()
  random = ra.random()

  if random < p_mut:
    # encontrar posiciones de los 1 y 0
    ones_pos = np.where(crom_mutado == 1)[0]
    zeros_pos = np.where(crom_mutado == 0)[0]
    # seleccionar un 1 y un 0 random
    swap_one = np.random.choice(ones_pos)
    swap_zero = np.random.choice(zeros_pos)
    # intercambiar los valores
    crom_mutado[swap_one], crom_mutado[swap_zero] = crom_mutado[swap_zero], crom_mutado[swap_one]

  return crom_mutado #devolver el array resultante


def validar(cromosoma, quorum_minimo):
  '''
  Validar que los datos cumplan con el quorum minimo.
  Si es mayor al quorum convierte unos random hasta cumplir
  Si es menor al quorum convierte ceros random hasta cumplir
  '''
  suma = np.sum(cromosoma)
  if suma > quorum_minimo:
    # posiciones de todos los 1s
    posiciones = np.where(cromosoma == 1)[0]
    k = suma - quorum_minimo
  elif suma < quorum_minimo:
    # posiciones de todos los 0s
    posiciones = np.where(cromosoma == 0)[0]
    k = quorum_minimo - suma
  else:
    return cromosoma
  
  # cambiar k bits aleatorios
  if k > 0:
    cambios = np.random.permutation(posiciones)[:k]
    cromosoma[cambios] ^= 1

  return cromosoma


def generarPoblacion(pobl_actual, quorum_min, p_mut, aProbs):
  ''' Obtiene dos padres aleatorios de acuerdo a la distribucion de probabilidad 
  acumulada, muta y valida los hijos obtenidos y devuelve la nueva poblacion'''

  n_crom, n_votos = pobl_actual.shape #filas y columnas de la poblacion

  nueva_poblacion = []

  #crear nueva poblacion con tamano n_crom - 1 (para agregar el mejor al final)
  while len(nueva_poblacion) < n_crom-1:
    # seleccionar padres usando choice() con probabilidades individuales
    i_padre1, i_padre2 = rng.choice(n_crom, size=2, replace=True, p=aProbs)

    p_corte = ra.randrange(n_votos) #punto de corte aleatorio
      
    #Cruzar los padres y obtener los hijos
    hijo1, hijo2 = crossover(pobl_actual[i_padre1], pobl_actual[i_padre2], p_corte)

    #validar hijos (ajustar hasta que cumplan el quorum minimo)
    hijo1_validado = validar(hijo1, quorum_min)
    hijo2_validado = validar(hijo2, quorum_min)

    #mutar hijos
    hijo1_mutado = mutacion(hijo1_validado, p_mut)
    hijo2_mutado = mutacion(hijo2_validado, p_mut)

    #agregar los hijos a la nueva poblacion
    nueva_poblacion.append(hijo1_mutado) 
    nueva_poblacion.append(hijo2_mutado)

  #agregar el mejor de la generacion anterior
  nueva_poblacion.append(pobl_actual[0])
  
  return np.array(nueva_poblacion) #retornar la nueva poblacion


#####################################################################
###########################-I-N-I-C-I-A-R-###########################
#####################################################################

puntos, n_votes = getDatos('./votes.json')    #posiciones y numero de votos
quorum_min = math.floor(n_votes/2)+1      #quorum minimo 
n_pobl = 37    # poblacion inicial
p_sel = 0.141     # prob seleccion del mejor fitness
p_mut = 0.1700019    # probailidad de mutacion
cont_gen = 1      # contador de generaciones
it_max = 25000    # numero de iteraciones maximas

probabilidades = prob_posicion(n_pobl, p_sel) #pesos de los cromosomas calculados a partir del mejor
matriz_distancias = distance_matrix(puntos, puntos) #matriz de distancias para O^n
mejor_dist_anterior = float('inf') # guardar mejor resultado anterior
start_time = time.time() #porsiaca

# generar la poblacion inicial 
poblacion_inicial = generarPoblacionInicial(n_pobl, n_votes, quorum_min) # generar la poblacion inicial
# ordenar la poblacion inicial de acuerdo al fitness
pobl_actual_ord = ordenarPoblacion(poblacion_inicial)

# definir un numero fijo de generaciones 
while cont_gen <= 25000: #mejor_dist_actual >= 9686.94:
  cont_gen += 1
  
  #generar la nueva poblacion (cruzar, validar, mutar)
  pobl_new = generarPoblacion(pobl_actual_ord, quorum_min, p_mut, probabilidades)

  #ordenar la nueva poblacion
  pobl_new_ord = ordenarPoblacion(pobl_new)

  # Guardar y graficar el mejor cromosoma de la nueva generacion
  mejor_cromosoma = pobl_new_ord[0]

  # Se reemplaza la poblacion anterior por la nueva
  pobl_actual_ord = pobl_new_ord

#TODO: mostrar tabla de datos y grafico descendente (generacion, Z)
# crear tupla global que guarde generacion y Z?
# crear tabla con pandas?
fig, ax = plt.subplots(figsize=(10, 9))  # crear figura del plt
graf_polig_convexo(mejor_cromosoma, puntos, ax)