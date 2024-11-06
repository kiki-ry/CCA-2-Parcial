import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF

#### 1. Importación de Datos: Importe el conjunto de datos y explore sus estadísticas.
carseats = pd.read_csv("Carseats(in).csv",header = 0)
with open('Carseats(in).csv', 'r') as file:
    lines = [line.strip().split(',') for line in file]
carseats = pd.DataFrame(lines[1:], columns = lines[0])
carseats = carseats.replace('"', '', regex = True)
carseats.columns = carseats.columns.str.replace('"', '', regex = True)

#print(carseats)

carseats['ShelveLoc'] = carseats['ShelveLoc'].map({'Bad': -1, 'Medium': 0, 'Good': 1}) # modifico los datos categóricos a numéricos
carseats['Urban'] = carseats['Urban'].map({'Yes': 1, 'No': 0}) # modifico los datos categóricos a numéricos
carseats['US'] = carseats['US'].map({'Yes': 1, 'No': 0}) # modifico los datos categóricos a numéricos

carseats = carseats.apply(pd.to_numeric, errors = 'coerce') # cambié a valores numéricos

#print(carseats.dtypes)

np.random.seed(0) #Establecemos una semilla aleatoria para garantizar la reproducibilidad de los resultados

Income = carseats["Income"].values.reshape(-1, 1) # variable Income como predictor
Sales = carseats["Sales"] # variable sales como variable de respuesta

Income_train, Income_test, Sales_train, Sales_test = train_test_split(Income, Sales, test_size = 0.25)

# Ajustamos un modelo de regresión lineal al conjunto de entrenamiento
model = LinearRegression()
model.fit(Income_train, Sales_train)

# Predecimos el valor medio de las viviendas para un valor de Income dado
Income_new = np.array([[15], [30], [60], [120]])
y_pred = model.predict(Income_new)

print(y_pred)

# Predicciones con coeficientes de la regresión
#y_pred_formula = (Income_train * model.coef_ + model.intercept_).reshape(Income_train.shape[0], )
#print('Predicciones 2:', y_pred_formula) #deberían darme iguales
#print(carseats)

# Ajustamos un modelo OLS con statsmodels para obtener los valores p y la estadística t
Income_train_sm = sm.add_constant(Income_train)
model_sm = sm.OLS(Sales_train, Income_train_sm).fit()
print(model_sm.summary())

###### 3. Coeficientes y Evaluación: Determine los coeficientes y evalúe los resultados
print("Coeficientes del modelo:", model.coef_)
print('El valor de "m" es igual a {0} y el de "b" a {1}.\nEntonces, y = {0} X + {1}.'.format(model.coef_.item(), model.intercept_))

mse = mse(y_pred)
r2 = r2_score(y_pred)
print("Error cuadrático medio:", mse)
print("Coeficiente de determinación R^2:", r2)