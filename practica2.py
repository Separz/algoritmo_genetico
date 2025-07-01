import json
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist

def getData(path):
  '''obtener los datos del json'''
  # cargar el archivo JSON
  with open(path) as f:
    data = json.load(f)
  
  # inicializar arrays
  ids = []        # identificador
  points = []     # lista de puntos [x, y]
  
  # extraer datos relevantes
  for rollcall in data['rollcalls']:
    for vote in rollcall['votes']:
      ids.append(vote['icpsr'])
      points.append([vote['x'], vote['y']])  # guardar [x,y] 
  
  nvotes = len(ids)
  return ids, np.array(points), nvotes  # Convertir a array numpy

#Funcion calcula el fitness sin librerias externas
#def fitness(chrom, puntos):
#  '''
#    Evaluar la suma de distancias a pares, a menor distancia mejor fitness
#  '''
#  punto_seleccionado = puntos[chrom == 1]  #solo puntos en coalicion
#  n = len(punto_seleccionado) # tamano de la lista
#  distancia_total = 0.0
#  for i in range(n): #recorrer x
#    for j in range(i + 1, n):  # Evita duplicados (i < j)
#      dx = punto_seleccionado[i][0] - punto_seleccionado[j][0]
#      dy = punto_seleccionado[i][1] - punto_seleccionado[j][1]
#      distancia_total += np.sqrt(dx**2 + dy**2)  # Distancia euclidiana
#  print(distancia_total)
#  return distancia_total

#utiliza libreria scipy y normaliza el resultado de 0 a 1
def fitness(chrom, puntos):
    selected_points = puntos[chrom == 1]
    n = len(selected_points)
    total_distance = np.sum(pdist(selected_points, 'euclidean')) #distancia euclidiana
    
    # Normalizar dividiendo por la maxima distancia posible (diagonal del bounding box)
    #max_possible_distance = np.sqrt(2) * len(selected_points)**2 
    #normalized_distance = total_distance / max_possible_distance # distancia normalizada entre [0, 1]
    
    # Máximo teórico para n puntos en [0,1]x[0,1]
    max_theoretical = (n * (n - 1) / 2) * np.sqrt(2)
    
    normalized_distance = total_distance / max_theoretical  # Ahora está en [0, 1]
    print(normalized_distance)
    return normalized_distance



def mutacion():
  '''cambiar un dato aleatorio de un cromosoma mediante una probabilidad'''
  ...

def ordenar():
  '''
    ordenar los individuos del mejor al peor fitness
  '''
  ...

def crossover():
  '''crea dos cromosomas hijos a partir de dos cromosomas padres
  necesita de un punto de corte para realizar el entrecruzamiento'''
  ...

def tournament_selection():
  ...

def genCrom(longitud, unos):
  '''genera un cromosoma que cumple con la coalicion minima ganadora'''
  # vector ceros
  cromosoma = np.zeros(longitud, dtype=int)
  # seleccionar posiciones aleatorias para los unos
  posicion = np.random.choice(longitud, size=unos, replace=False)
  # asignar unos a las posiciones
  cromosoma[posicion] = 1
  
  return cromosoma

def genPob(pob, n_votes, quorum_min):
  poblacion = np.array([])
  for i in range(0, pob, 1):
    cromosoma = genCrom(n_votes, quorum_min)
    poblacion = np.append(poblacion, cromosoma, axis=0)
    print(poblacion)


ids, puntos, n_votes = getData('./votes.json')    #id y posiciones
quorum_min = math.floor(n_votes/2)+1      #quorum minimo
pobl = 10
#crom = genCrom(n_votes, quorum_min) #generar cromosoma



genPob(pobl, n_votes, quorum_min)

#fitness(crom, puntos)

#poblacion es una matriz con filas correspondientes a los cromosomas



#PASOS:
#1.-Crear la poblacion inicial tamano pobl maxima -> luego de la primera iteracion debe ser impar
#2.-Evaluar cada cromosoma de acuerdo a Z(fitness) -> que tan bueno es
#3.-Ordenar la poblacion de manera decendente de acuerdo al fitness
#4.-Seleccionar los padres (Dos cromosomas)
#5.-Seleccionar punto de corte
#6.-Cruzar padres
#7.-Generar mutaciones con probabilidad P_mut
#8.-Validar hijos (se refiere a que cumplan con el quorum?)
#9.-Poblar la nueva poblacion con los hijos supervivientes
#9 -> 2 pasar el mejor de la poblacion anterior
#9 -> 4 hasta llenar la poblacion





#torneo p -> probabilidad de seleccionar al mejor
#combinacion convexa todo suma 1
#p
#p(1-p) / sumatoria p(1-p)**i-1
#p(1-p)**2 / sumatoria p(1-p)**i-1
#p(1-p)i-1 / sumatoria p(1-p)**i-1
#si no suma 1 se debe normalizar


#P/sum p(1-p)**i-1
#P+p(1-p) / sum p(1-p)i-1
#p+p(1-p)+p(1-p)**2 / sum p(1-p)i-1