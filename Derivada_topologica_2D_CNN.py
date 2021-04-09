import tensorflow as tf 
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
#Módulo para importar la base de datos propia
from LoadData import loadData

# Importar el set de datos de nuestra base propia: ############################
'''Especificar el nombre de la carpeta de la base de datos:'''
DatabaseName0="FinalDatabase30"
DatabaseName01="FinalDatabase301"
DatabaseName02="FinalDatabase302"
DatabaseName03="FinalDatabase303"
DatabaseName04="FinalDatabase304"
DatabaseName05="FinalDatabase305"
''' En esta variación utilizamos DTt solo '''
''' ================================================================ '''
[data0a,data0b,data0c]=loadData(DatabaseName0)
train_experiments0=[]; train_solutions0=[]
test_experiments0=[]; test_solutions0=[]
validation_experiments0=[]; validation_solutions0=[]
''' ================================================================ '''
[data01a,data01b,data01c]=loadData(DatabaseName01)
train_experiments01=[]; train_solutions01=[]
test_experiments01=[]; test_solutions01=[]
validation_experiments01=[]; validation_solutions01=[]
''' ================================================================ '''
[data02a,data02b,data02c]=loadData(DatabaseName02)
train_experiments02=[]; train_solutions02=[]
test_experiments02=[]; test_solutions02=[]
validation_experiments02=[]; validation_solutions02=[]
''' ================================================================ '''
[data03a,data03b,data03c]=loadData(DatabaseName03)
train_experiments03=[]; train_solutions03=[]
test_experiments03=[]; test_solutions03=[]
validation_experiments03=[]; validation_solutions03=[]
''' ================================================================ '''
[data04a,data04b,data04c]=loadData(DatabaseName04)
train_experiments04=[]; train_solutions04=[]
test_experiments04=[]; test_solutions04=[]
validation_experiments04=[]; validation_solutions04=[]
''' ================================================================ '''
[data05a,data05b,data05c]=loadData(DatabaseName05)
train_experiments05=[]; train_solutions05=[]
test_experiments05=[]; test_solutions05=[]
validation_experiments05=[]; validation_solutions05=[]

#Extraer los datos training y sus soluciones 30:
for i in range(len(data0a)):
    train_experiments0.append(data0a[i][0])
    train_solutions0.append(data0a[i][1])
#Los test:
for i in range(len(data0b)):
    test_experiments0.append(data0b[i][0])
    test_solutions0.append(data0b[i][1])
#Los datos de validación:
for i in range(len(data0c)):
    validation_experiments0.append(data0c[i][0])
    validation_solutions0.append(data0c[i][1])
#··············································································
#Extraer los datos training y sus soluciones 301:
for i in range(len(data01a)):
    train_experiments01.append(data01a[i][0])
    train_solutions01.append(data01a[i][1])
#Los test:
for i in range(len(data01b)):
    test_experiments01.append(data01b[i][0])
    test_solutions01.append(data01b[i][1])
#Los datos de validación:
for i in range(len(data01c)):
    validation_experiments01.append(data01c[i][0])
    validation_solutions01.append(data01c[i][1])
#··············································································
#Extraer los datos training y sus soluciones 302:
for i in range(len(data02a)):
    train_experiments02.append(data02a[i][0])
    train_solutions02.append(data02a[i][1])
#Los test:
for i in range(len(data02b)):
    test_experiments02.append(data02b[i][0])
    test_solutions02.append(data02b[i][1])
#Los datos de validación:
for i in range(len(data02c)):
    validation_experiments02.append(data02c[i][0])
    validation_solutions02.append(data02c[i][1])
#··············································································
#Extraer los datos training y sus soluciones 303:
for i in range(len(data03a)):
    train_experiments03.append(data03a[i][0])
    train_solutions03.append(data03a[i][1])
#Los test:
for i in range(len(data03b)):
    test_experiments03.append(data03b[i][0])
    test_solutions03.append(data03b[i][1])
#Los datos de validación:
for i in range(len(data03c)):
    validation_experiments03.append(data03c[i][0])
    validation_solutions03.append(data03c[i][1])
#··············································································
#Extraer los datos training y sus soluciones 304:
for i in range(len(data04a)):
    train_experiments04.append(data04a[i][0])
    train_solutions04.append(data04a[i][1])
#Los test:
for i in range(len(data04b)):
    test_experiments04.append(data04b[i][0])
    test_solutions04.append(data04b[i][1])
#Los datos de validación:
for i in range(len(data04c)):
    validation_experiments04.append(data04c[i][0])
    validation_solutions04.append(data04c[i][1])
#··············································································
#Extraer los datos training y sus soluciones 305:
for i in range(len(data05a)):
    train_experiments05.append(data05a[i][0])
    train_solutions05.append(data05a[i][1])
#Los test:
for i in range(len(data05b)):
    test_experiments05.append(data05b[i][0])
    test_solutions05.append(data05b[i][1])
#Los datos de validación:
for i in range(len(data05c)):
    validation_experiments05.append(data05c[i][0])
    validation_solutions05.append(data05c[i][1])





train_experiments = train_experiments0+train_experiments01+train_experiments02+train_experiments03+train_experiments04+train_experiments05

train_solutions = train_solutions0+train_solutions01+train_solutions02+train_solutions03+train_solutions04+train_solutions05

test_experiments = test_experiments0+test_experiments01+test_experiments02+test_experiments03+test_experiments04+test_experiments05

test_solutions = test_solutions0+test_solutions01+test_solutions02+test_solutions03+test_solutions04+test_solutions05

validation_experiments = validation_experiments0+validation_experiments01+validation_experiments02+validation_experiments03+validation_experiments04+validation_experiments05

validation_solutions = validation_solutions0+validation_solutions01+validation_solutions02+validation_solutions03+validation_solutions04+validation_solutions05

experiments = train_experiments + test_experiments + validation_experiments
experiments = np.array(experiments)
total_experiments = experiments+1
total_experiments = np.reshape(total_experiments,(total_experiments.shape[0],10,10,1))
print(total_experiments.shape)
solutions = train_solutions + test_solutions + validation_solutions
total_solutions = np.array(solutions)*100
print(total_solutions.shape)


numeros = list(range(0,29900))
rand.shuffle(numeros)
print(type(numeros))
print(len(numeros)-len(set(numeros)))  #para comprobar si se ha repetido algun numero, si sale 0, no se ha repetido
numeros_random = np.array(numeros) #pasar  la lista a vector
print(numeros_random.shape)

train_experiments_final = total_experiments[numeros_random[0:23000],:,:]
test_experiments_final = total_experiments[numeros_random[23000:28800],:,:]
validation_experiments_final = total_experiments[numeros_random[28800:29900],:,:]

train_solutions_final = total_solutions[numeros_random[0:23000]]
test_solutions_final = total_solutions[numeros_random[23000:28800]]
validation_solutions_final = total_solutions[numeros_random[28800:29900]]

print(train_experiments_final.shape,
      test_experiments_final.shape,
      validation_experiments_final.shape,
      train_solutions_final.shape,
      test_solutions_final.shape,
      validation_solutions_final.shape)



#vemos como se muestran las termografias con el mismo defecto pero diferente ruido
import matplotlib 
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
def scatter3d(x,y,z, cs, colorsMap = 'jet'):
       cm = plt.get_cmap(colorsMap)
       cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax = max(cs))
       scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
       fig = plt.figure()
       ax = Axes3D(fig)
       ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
       scalarMap.set_array(cs)
       fig.colorbar(scalarMap)
       plt.show()



train_experiments_final_imagen=np.reshape(train_experiments_final,(train_experiments_final.shape[0],10,10))

import numpy as np
plt.imshow(np.asarray(train_experiments_final_imagen[0]),cmap='plasma')
plt.colorbar()
plt.show()


for i in range (0,15):
    plt.subplot(3,5,i+1)
    plt.imshow(train_experiments_final_imagen[i], cmap='plasma')
    plt.axis('off')
plt.show()  


    
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3,3),padding='same',input_shape=train_experiments_final[0].shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, kernel_size=(3,3),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(2048,kernel_initializer='random_normal',bias_initializer='zeros', use_bias=True), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(1024, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(512, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(256, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(128, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(64, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(32, use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)])

from tensorflow.keras.callbacks import LearningRateScheduler
import math
# Compilar el modelo __________________________________________________________
opt=tf.keras.optimizers.SGD(momentum=0.9, nesterov=True, name='SGD') #momento + Nesterov
model.compile(
    optimizer=opt,
    loss='mse',
    metrics=['mae']) 

# Definimos como va a ser el ratio de aprendizaje exponencial.
initial_learning_rate = 0.1
def lr_exp_decay(epoch, learning_rate):
    k = 0.4
    return initial_learning_rate * math.exp(-k*epoch)  #Lr exponencial

# Ajustamos el modelo.
training_history = model.fit(
        train_experiments_final, 
        train_solutions_final, 
        batch_size=20,
        epochs=20, 
        validation_data=(test_experiments_final, test_solutions_final),
        callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)]
        )

############################################################################
# Pintamos el modelo
model.summary() 
pd.DataFrame(training_history.history).plot(figsize=(12,7))
plt.grid(True)
plt.show()

############

s = np.argsort(validation_solutions_final)
print(s)
validation_solutions_final_ordenado = np.sort(validation_solutions_final)
print(np.argsort(validation_solutions_final_ordenado))

validation_experiments_final_ordenado = validation_experiments_final[validation_solutions_final.argsort()]
print(validation_experiments_final_ordenado.shape)

#media movil
def sma(data, period):   #subfuncion para obtener el sma
    sma1=np.zeros(data.size-50)
    for step in range(len(sma1)):
        sma1[step]=np.mean(data[step-50:step+50])
    return sma1
periodo = 100

# Comparamos las predicciones con las originales
from sklearn.linear_model import LinearRegression
predictions = model.predict(validation_experiments_final_ordenado)
x=np.zeros(1100)
y=np.zeros(1100)
z=np.zeros(1100)
for k in range (1):
    for i in range (0,1100):
      x[i]=i
    for i in range (0,1100):
      y[i]=predictions[i] 
    for i in range (0,1100):
      z[i]=validation_solutions_final_ordenado[i]
    sma_y = sma((y),periodo)
    sma_error = sma(abs(y-z),periodo)
    regr = LinearRegression()
    regr1 = LinearRegression()
    xr = x[:, np.newaxis]
    regr.fit(xr,y)
    regr1.fit(xr,abs(y-z))
    fig, ax = plt.subplots(1,figsize = (24,10))
    fig.suptitle("Derivada topológica: Modelo 6", fontsize=40)
    ax.plot(sma_y, color = 'red', linewidth=5, label="Media móvil 100")
    ax.plot(sma_error, color = 'salmon', linewidth=5, label="Media móvil 100")
    ax.plot(xr,regr.predict(xr), color = 'royalblue', linewidth=5, label="Regresión lineal")
    #ax.plot(xr,regr1.predict(xr), color = 'cyan', linewidth=5, label="Regresión lineal")
    plt.scatter(x,y,label="Predicción",s=15)
    plt.scatter(x,z,label="Valor real",s=15, color="brown")
    plt.bar(x,abs(y-z), label="Error absoluto", color="green")
    ax.legend(loc="upper left", frameon=False, fontsize=22 )
    plt.xlabel("Experimentos",fontsize=30)
    plt.ylabel("Profundidad relativa", fontsize=25)
    plt.xticks(size="20")
    plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),size="20")
    plt.show()



############################################################################################
#np.savetxt('media_movil_100_error_6.csv',sma_error,delimiter=';')
#np.savetxt('media_movil_100_prediccion_6.csv',sma_y,delimiter=';')


top_layer = model.layers[0]
for i in range (1,65):
    plt.subplot(8,8,i)
    plt.imshow(top_layer.get_weights()[0][:, :, :, i-1].squeeze(), cmap='plasma', vmax=1.5, vmin=-1.5 )
    plt.axis('off')
plt.show()  
    
plt.imshow(top_layer.get_weights()[0][:, :, :, 60].squeeze(), cmap='plasma', vmax=1.5, vmin=-1.5)
plt.colorbar()
plt.show()
