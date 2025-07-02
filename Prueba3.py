import json
import math
import numpy as np
import matplotlib.pyplot as plt
import random as ra
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

np.set_printoptions(edgeitems=10, threshold=100, linewidth=150)
# Configurar el generador de numeros aleatorios
rng = np.random.default_rng()

'''
Dudas:
-La mutacion? ya que el cruce al ser random rompe el quorum, y se tiene una 
funcion validar que ajusta los votos hasta que se cumpla el quorum.
()->
-probabilidad de mutar por elemento del cromosoma o por cromosoma?
-ejecutar hasta que el fitness no mejore por x generaciones?
-guardar todas las generaciones para encontrar el minimo local?
-la probabilidad de seleccionar al mejor es seteada por el usuario o se calcula a partir del fitness?
-los padres se seleccionan mediante la probabilidad acumlada, el mejor pasa directamente?
-los padres deben ser diferentes'''

#mutacion 1 si es 0 y 0 si es 1, solo 1 dato
#el Z no deberia subir solo bajar
#tupla <it, z> -> iteracion, distancia, solo imrimir si hay cmabios
#numpy.random.generator.choice -> para elegir el padre con reemplazo
#^ pasar el arreglo del seleccion p que suma casi 1 luego lo normaliza la funcion
#mostrar el grafico al final
#notas:
# -tratar de usar numpy para calculos mas rapido
# -

#cambiar:
# el erro debe estar en la posicion de validar luego de la mutacion
# mutacion debe cambiarse
# eleccion de padres se debe usar np.random.choice()-> pasar el arreglo de probabilidades
# 
#
#-

def getDatos(path):
  '''obtener los datos del json'''
  # cargar el archivo JSON
  with open(path) as f:
    data = json.load(f)

  # inicializar arrays
  ids = []         # identificador
  points = []      # lista de puntos [x, y]

  # extraer datos relevantes
  for rollcall in data['rollcalls']:
    for vote in rollcall['votes']:
      ids.append(vote['icpsr'])
      points.append([vote['x'], vote['y']])  # guardar [x,y] 

  n_votes = len(ids)
  return np.array(ids), np.array(points), n_votes  # convertir a array np


def generarCromosoma(total, quorum):
  '''genera un cromosoma que cumple con la coalicion minima ganadora'''
  # vector ceros
  cromosoma = np.zeros(total, dtype=np.uint8)
  # seleccionar posiciones aleatorias para los unos
  posicion = np.random.choice(total, size=quorum, replace=False)
  # asignar unos a las posiciones
  cromosoma[posicion] = 1

  return cromosoma


def graf_polig_convexo(cromosoma, pts, ax):
    '''crea el grafico interactivo para mostrar los puntos en cada generacion'''
    ax.clear()  # limpiar el grafico anterior
    
    # filtrar puntos
    pts_select = pts[cromosoma == 1]
    
    # graficar puntos
    ax.scatter(pts[:, 0], pts[:, 1], color='blue', label='Todos los puntos')
    ax.scatter(pts_select[:, 0], pts_select[:, 1], color='red', label='Puntos seleccionados')
    
    # calcular y graficar el poligono convexo
    if len(pts_select) >= 3:
        hull = ConvexHull(pts_select)
        # dibujar aristas
        for simplex in hull.simplices:
            ax.plot(pts_select[simplex, 0], pts_select[simplex, 1], 'c-')
        # dibujar vertices
        ax.plot(pts_select[hull.vertices, 0], pts_select[hull.vertices, 1], 'o',
                mec='orange', color='none', lw=1, markersize=8)
    else:
        print("No hay puntos suficientes para formar el poligono convexo")
    
    ax.set_xlabel('Posición Política')
    ax.set_ylabel('Posición Política (otro)')
    ax.set_title('Algoritmo Genético para MWC')
    ax.grid(True)
    ax.axis([-1, 1, -1, 1])
    ax.legend()
    plt.draw()  # actualizar el grafico
    plt.pause(0.1)  # tiempo de pausa para actualizar


def generarPoblacionInicial(n_pob, n_votes, quorum):
  poblacion = np.empty((n_pob, n_votes), dtype=np.uint8) # crear vacio
  for i in range(n_pob):
    poblacion[i] = generarCromosoma(n_votes, quorum)
  return poblacion


#utiliza scipy pdist y normaliza el resultado de 0 a 1
def calcularFitness(cromosoma, puntos):
  """
  fitness normalizado 0 a 1 para puntos en el plano
  1 Minima distancia; 0 Maxima distancia posible.
  """
  pts_selec = puntos[cromosoma == 1]

  p_max = len(pts_selec) #216 puntos
    
  #suma de distancia total
  distancia_total = np.sum(pdist(pts_selec, 'euclidean'))
    
  #distancia maxima en el plano -1,-1 a 1,1
  d_max = np.sqrt((1-(-1))**2+((1-(-1))**2)) #sqrt(8)
  max_possible_dist = (p_max * (p_max - 1) / 2) * d_max

  #fitness normalizado para minimizacion
  fitness = 1 - (distancia_total / max_possible_dist)
  
  return fitness

def distanciaTotal(cromosoma, puntos):
  pts_selec = puntos[cromosoma == 1]
  #suma de distancia total
  distancia_total = np.sum(pdist(pts_selec, 'euclidean'))
  return distancia_total


def ordenarPoblacion(poblacion, puntos):
  '''Se selecciona al mejor individuo de acuerdo a la probabilidad del fitness '''

  # 2.-Evaluar cada cromosoma de acuerdo a Z(fitness) 
  fitness_values = np.array([calcularFitness(cromosoma, puntos) for cromosoma in poblacion])
  distancias_totales = np.array([distanciaTotal(cromosoma, puntos) for cromosoma in poblacion])

  # 3.-Ordenar la poblacion de manera decendente de acuerdo al fitness
  sorted_indices = np.argsort(fitness_values)[::-1]
  
  # poblacion y valores ordenados
  poblacion_ordenada = poblacion[sorted_indices]
  fitness_ordenado = fitness_values[sorted_indices]
  distancias_ordenadas = distancias_totales[sorted_indices]

  # imprimir el mejor real
  print(f'Mejor fitness: {fitness_ordenado[0]}')
  print(f'Mejor distancia: {distancias_ordenadas[0]}')

  return poblacion_ordenada



def prob_posicion(poblacion, prob):
  '''calcular la probabilidad de elegir al individuo segun la posicion 
  en el indice retorna el arreglo de probabilidades acumuladas'''
  sum = 0
  probabilidades = []
  probabilidad = 0

  n_crom, n_votes = poblacion.shape

  #calcula la sumatoria de todas las probabilidades
  for j in range(1, n_crom+1):
    sum += prob*(1-prob)**(j-1)

  #calcula la probabilidad por posicion
  for i in range(1, n_crom+1):
    # calcular cuantil de acuerdo a la probabilidad de seleccion del mejor
    probabilidad = prob*(1-prob)**(i-1)/sum
    probabilidades.append(probabilidad) # guardar

  return probabilidades


def generarPoblacion(pobl_actual, p_sel, quorum_min, p_mut):
  '''Parametros: 
    - poblacion actual
    - prob de seleccionar padres
    - coalicion minima
    - prob de mutacion
  Obtiene dos padres aleatorios de acuerdo a la distribucion de probabilidad acumulada
  Muta y valida los hijos obtenidos y devuelve la nueva poblacion'''

  n_crom, n_votos = pobl_actual.shape
  probabilidades = prob_posicion(pobl_actual, p_sel)

  #seleccionar 2 padres de manera aleatoria de acuerdo la probabilidad por cuantil
  nueva_poblacion = []
  #crear nueva poblacion con tamano n_crom - 1 (para agregar el mejor al final)
  while len(nueva_poblacion) < n_crom-1:
    # seleccionar padres usando choice() con probabilidades individuales
    i_padre1, i_padre2 = rng.choice(n_crom, size=2, replace=True, p=probabilidades)

    p_corte = ra.randrange(n_votos) #random para el par de cromosomas
      
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
  mejor = pobl_actual[0]
  nueva_poblacion.append(mejor)
  #print(f"Tamano nueva pobl: {len(nueva_poblacion)}")
  return np.array(nueva_poblacion) #retornar la nueva poblacion


def crossover(padre1, padre2, p_corte):
  '''Cruzar dos pares de cromosomas dado un punto de corte
  Devuelve dos cromosomas hijos'''

  hijo1 = np.concatenate((padre1[:p_corte], padre2[p_corte:]))
  hijo2 = np.concatenate((padre2[:p_corte], padre1[p_corte:]))

  return hijo1, hijo2


def mutacion(crom_original, p_mut):
  '''cambiar un dato aleatorio de un cromosoma mediante la probabilidad p_mut'''
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
  #devolver el array resultante
  return crom_mutado

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


#########---INICIO---#########

ids, puntos, n_votes = getDatos('./votes.json')    #id y posiciones
quorum_min = math.floor(n_votes/2)+1      #quorum minimo
n_pobl = 38    # poblacion inicial
p_sel = 0.141     # prob seleccion del mejor finess
p_mut = 0.1700019    # probailidad de mutacion
cont_gen = 1  # contador de generaciones
pobl_all = [] # guardar todas las poblaciones?

fig, ax = plt.subplots()  # crear figura del plt

# generar la poblacion inicial 
poblacion_inicial = generarPoblacionInicial(n_pobl, n_votes, quorum_min) # generar la poblacion inicial
# ordenar la poblacion de acuerdo al fitness
pobl_actual_ord = ordenarPoblacion(poblacion_inicial, puntos)

#graficar el mejor cromosoma de la poblacion inicial
mejor_cromosoma = pobl_actual_ord[0]
graf_polig_convexo(mejor_cromosoma, puntos, ax)
    

# definir un numero fijo de generaciones 
while True:
    print(f'Generacion Actual: {cont_gen}')
    cont_gen += 1
    #generar la nueva poblacion (cruzar, mutar, validar)
    pobl_new = generarPoblacion(pobl_actual_ord, p_sel, quorum_min, p_mut)
    #ordenar la nueva poblacion
    pobl_new_ord = ordenarPoblacion(pobl_new, puntos)
    
    # Guardar y graficar el mejor cromosoma de la nueva generacion
    mejor_cromosoma = pobl_new_ord[0]
    graf_polig_convexo(mejor_cromosoma, puntos, ax)
    
    # Se reemplaza la poblacion anterior por la nueva
    pobl_actual_ord = pobl_new_ord