# ProyectoFinal

# Maestrante: Renato Jácome
----------------------------------------------
# Se explora y se importa el dataset de la url: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
# Se familiariza con las características

id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \
        
0    842302         M        17.99         10.38          122.80     1001.0  

1    842517         M        20.57         17.77          132.90     1326.0 

2  84300903         M        19.69         21.25          130.00     1203.0 

3  84348301         M        11.42         20.38           77.58      386.1

4  84358402         M        20.29         14.34          135.10     1297.0 

-------------------------------------------------------------

# Verificar valores nulos dentro del dataset - El dataset no presenta valores nulos
diagnosis                  0

radius_mean                0

texture_mean               0

perimeter_mean             0

-------------------------------------------------------------

# Verificar valores inconsistentes 
radius_mean  texture_mean  perimeter_mean    area_mean  \
count   569.000000    569.000000      569.000000   569.000000

mean     14.127292     19.289649       91.969033   654.889104 

std       3.524049      4.301036       24.298981   351.914129 

min       6.981000      9.710000       43.790000   143.500000

25%      11.700000     16.170000       75.170000   420.300000

50%      13.370000     18.840000       86.240000   551.100000

75%      15.780000     21.800000      104.100000   782.700000

max      28.110000     39.280000      188.500000  2501.000000

-------------------------------------------------------------

# Utilizo la característica de diagnóstico "diagnosis" de mayor relevancia, convierto un string en entero para facilitar la interpretación y conformo los conjuntos de entrenamiento y prueba

Dimensiones del conjunto de entrenamiento: (455, 31)

Dimensiones del conjunto de prueba: (114, 31)

-------------------------------------------------------------

# Función para mostrar la matriz de confusión que será usada en los diferentes modelos
def plot_confusion_matrix(y_true, y_pred, title):

    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    plt.show()

-------------------------------------------------------------

# Pruebo y evalúo los siguientes modelos:
    Regresión Logística (logistic_regression)
    Exactitud del modelo: 0.6228070175438597
    
    Bosques Aleatorios (Random Forest)
    Exactitud del modelo: 0.9649122807017544
    
    Clasificación-regresión - Máquinas de Vector Soporte (SVM)
    Exactitud del modelo: 0.6228070175438597

-------------------------------------------------------------

# Existe una correlación entre las matrices de confusión de training y test, lo que significa que generaliza bien los datos y no existe un sobreajuste.

-------------------------------------------------------------

# El mejor modelo que se adapata a los resultados es el modelo de regresión logística con los siguientes resultados:

Matriz de confusión:

[[70  1]

 [ 4 39]]

Como resultado 70 verdaderos positivos, 1 falso negativo, 4 falsos negativos y 39 verdaderos negativos. Con una exactitud de modelo de 0.956140350877193

No opté por el modelo random forest a pesar de tener mayor exactitud 0.9649122807017544; ya que existe una diferencia notable entre la matriz de training y test lo que significa que generaliza mal los datos y existe sobreajuste.

-------------------------------------------------------------
