#!/usr/bin/env python
# coding: utf-8

# ###  Caso de Obesidad
# ##### Universidad Externado
# ##### Seminario de Programación
# ##### Anjely Amazo, Liliana Hernandez y Julieth Perea
# 

# #  Nivel de obesidad

# En el siguiente enlace encontrarán una base de datos que permiten estimarlos niveles de obesidad en individuos de los países de México, Perú y Colombia, con base en sus hábitos alimenticios y condición física. Los datos contienen 2111 registros, además, hay una variable `NObesidad` (Nivel de obesidad), que permite clasificar los datos utilizando los valores de Peso Insuficiente, Peso Normal, Nivel de Sobrepeso I, Nivel de Sobrepeso II, Obesidad Tipo I , Obesidad tipo II y Obesidad tipo III. El 77% de los datos se generaron sintéticamente utilizando la herramienta Weka y el filtro SMOTE, el 23% de los datos se recopilaron directamente de los usuarios a través de una plataforma web, la información acerca de los datos y los paper relevantes que se han creado a partir de este ejercicio lo pueden encontrar en:
# 
# [ Estimation of obesity levels based on eating habits and physical condition Data Set ](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+)
# 
# 

# ### ¿Como afectan la condición física y los hábitos alimenticios a la obesidad?

# Iniciamos con un analisis exploratorio de la base, cargando la base y las librerias: 

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#DataO=pd.read_csv('../data/ObesityDataSet_raw_and_data_sinthetic.csv')
DataO=pd.read_csv('C:/Users/perea/OneDrive/Escritorio/Taller obesidad/ObesityDataSet_raw_and_data_sinthetic.csv')


# Evidenciamos las variables que trae la base las cuales son 16, y vienen de tipo númerico y caracteres:

# In[47]:


DataO.info()


# En primer lugar, ajustaremos la base de datos a explotar y ajustamos los nombres de las variables de acuerdo a su información

# 

# In[48]:


DataO.set_axis(['Genero', 'Edad', 'Estatura','Peso','Familiar_tiene_sobrepeso','Consumo_frecuente_alimentos_altos_calorias','Frecuencia_consumo_verduras','Numero_comidas_principales','Consumo_alimentos_entre_comidas','Fuma','Consumo_agua_diaria','Monitoreo_calorias_diaria','Frecuencia_actividad_fisica','Tiempo_utilizando_tecnologia','Consumo_alcohol','Transporte_utilizado','Nivel_obesidad'], 
                 axis='columns', inplace=True)
DataO.info()


# De acuerdo a la información encontrada en el link de descarga, sabemos que tenemos los siguientes datos: 
# Consumo frecuente de alimentos altos en calorías (FAVC), Frecuencia de consumo de verduras (FCVC), Número de comidas principales (NCP), Consumo de alimentos entre comidas (CAEC), Consumo de agua diaria (CH20) y Consumo de alcohol (CALC). Los atributos relacionados con la condición física son: Monitoreo del consumo de calorías (SCC), Frecuencia de actividad física (FAF), Tiempo utilizando dispositivos tecnológicos (TUE), Transporte utilizado (MTRANS), otras variables obtenidas fueron: Género, Edad, Altura y Peso. Finalmente, se etiquetaron todos los datos y se creó la variable de clase NObesidad con los valores de: Peso Insuficiente, Peso Normal, Sobrepeso Nivel I, Sobrepeso Nivel II, Obesidad Tipo I, Obesidad Tipo II y Obesidad Tipo III, con base en la Ecuación (1) e información de la OMS y la Normatividad Mexicana. 

# In[49]:


#tipo de registro
display("Tipos de registros en el DataFrame",DataO.dtypes)


# En segundo lugar, realizaremos la manipulacion y tratamiento de los datos y redondeamos la base de datos de la siguiente manera:

# In[95]:


DataO["Edad"]=round(DataO["Edad"])
DataO["Frecuencia_consumo_verduras"]=round(DataO["Frecuencia_consumo_verduras"])
DataO["Numero_comidas_principales"]=round(DataO["Numero_comidas_principales"])
DataO["Consumo_agua_diaria"]=round(DataO["Consumo_agua_diaria"])
DataO["Frecuencia_actividad_fisica"]=round(DataO["Frecuencia_actividad_fisica"])
DataO["Tiempo_utilizando_tecnologia"]=round(DataO["Tiempo_utilizando_tecnologia"])


# El siguiente paso es convertir las variables de dummies: 

# In[96]:


DataO.Genero = DataO.Genero.replace({"Female": 1, "Male": 2})
DataO.Familiar_tiene_sobrepeso = DataO.Familiar_tiene_sobrepeso.replace({"yes": 1, "no": 2})
DataO.Consumo_frecuente_alimentos_altos_calorias = DataO.Consumo_frecuente_alimentos_altos_calorias.replace({"yes": 1, "no": 2})
DataO.Fuma = DataO.Fuma.replace({"yes": 1, "no": 2})
DataO.Monitoreo_calorias_diaria = DataO.Monitoreo_calorias_diaria.replace({"yes": 1, "no": 2})
DataO.Nivel_obesidad = DataO.Nivel_obesidad.replace({"Obesity_Type_I": 1, "Obesity_Type_II": 2,"Obesity_Type_III" :3 , "Overweight_Level_I" :4,"Overweight_Level_II" :5,"Normal_Weight" :6, "Insufficient_Weight" :7},)


# Ahora bien, se realiza la transformación al tipo de número:

# In[54]:


DataO.Genero = DataO.Genero.apply(int)
DataO.Edad = DataO.Edad.apply(int)
DataO.Nivel_obesidad = DataO.Nivel_obesidad.apply(int)


# A continuación, visualizamos la información tratada:

# In[55]:


DataO.info()


# In[57]:


DataO


# In[56]:


DataO.describe()


# ### Descripción Datos

# La información anterior, refleja que el registro corresponde a 1.068 al género masculino es decir el 50,59% y 1.043 al femenino con el 49.41%, como se ilustra a continuación:

# 

# In[60]:


genero_res= pd.DataFrame(DataO['Genero'].value_counts(sort=True))
genero_res


# In[61]:


valores = [1068, 1043]
nombres_orden = ["Male","Female"]
colores = ["#EE6055","#60D394"]
desfase = (0, 0)
plt.pie(valores, labels=nombres_orden, autopct="%0.2f %%", colors=colores, explode=desfase)
plt.axis("equal")
plt.show()


# Ahora bien, al analizar el nivel de obesidad se identificó que el tipo de obesidad se concentra en estos 3 niveles: la obesidad tipo I cuenta con 351 registros, seguido de la obesidad tipo III con 324 registros y la obesidad tipo II con el 297 de los casos.

# In[64]:


NObeyesdad_res= pd.DataFrame(DataO['Nivel_obesidad'].value_counts(sort=True))
NObeyesdad_res


# In[65]:


axis = NObeyesdad_res.plot.barh(rot=0)
print(axis)
plt.show()


# En este punto, analizamos la obesidad desde la variable consumo de alcohol y se identificó que los casos registrados 1.401 indicó que consumo a veces licor (Somethimes), es decir el 66%, en 639 casos no consumo licor (no) con un 30%, en 70 casos consumen frecuentemente (Frequently) con el 3% y 1 un caso indico que siempre, como se ilustra a continuación:

# In[77]:


Consumo_alcohol= pd.DataFrame(DataO['Consumo_alcohol'].value_counts(sort=True))
Consumo_alcohol


# In[78]:


etiquetas = ["Sometimes","no","Frequently", "Always"] 
porcentas = [1401,639,70,1]
plt.pie(porcentas,labels = etiquetas, radius = 2,
        startangle=10, autopct = '%1.1f%%')
plt.show()  


# ### Exploración de datos
# 

# Es muy importante identificar datos atípicos, nulos y relaciones entre variables (colinealidad, alta correlación, etc.).

# Por otra parte, realizamos un analisis de la correlación graficamente entre el peso y la estatura:

# Ahora revisemos la correlación de los datos:

# In[97]:


DataO.corr(method='pearson', 
               min_periods=1)


# In[115]:


corr = numericos_DataO.corr()
corr.style.background_gradient(cmap='coolwarm')


# Se analizó la correlación graficamente entre el peso y la estatura, y se ilustra lo siguiente:

# In[98]:


fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.scatter(x=DataO.Estatura, y=DataO.Peso, alpha= 0.8, color="purple")
ax.set_xlabel('Estatura')
ax.set_ylabel('Peso');


# In[85]:


pip install seaborn


# In[86]:


conda install seaborn


# In[87]:


import seaborn as sns
sns.scatterplot(data=DataO,x='Peso',y='Estatura',hue='Nivel_obesidad')


# De la gráfica anterior, se evidencia una relación entre la altura y el peso en el nivel de obesidad, en estge punto se creara la variable indice de masa muscular y se ilustra lo siguiente:

# In[136]:


DataO["Masa_Muscular"]= (DataO["Peso"])/((DataO["Estatura"])*(DataO["Estatura"]))


# In[137]:


DataO


# ## Exploración Previa - Solo cuantitativa

# 1. Tomen las variables cuantitativas y determinen agrupamientos en la base de datos.

# In[138]:


np.random.seed(2021)


# In[139]:


corr = numericos_DataO.corr()
corr.style.background_gradient(cmap='coolwarm')


# ### Modelo de clasificación con una red neuronal

# En este punto, se creara un modelo para predecir la variable nivel de obesidad (NObeyesdad)

# In[150]:


Data_exclusion = DataO.drop(['Nivel_obesidad'], axis=1)
Data_exclusion


# In[145]:


vb_x = Data_exclusion.drop('Nivel_obesidad', axis=1).values
vb_y = np.asarray(pd.get_dummies(Data_exclusion['Nivel_obesidad']).values).astype('float32')
print(vb_x.shape, vb_y.shape)


# In[ ]:


vb_x = df_final.drop('NObeyesdad', axis=1).values
vb_y = np.asarray(pd.get_dummies(df_final['NObeyesdad']).values).astype('float32')
vb_x;vb_y

