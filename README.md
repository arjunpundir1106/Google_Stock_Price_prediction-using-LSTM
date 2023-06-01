# Google_Stock_Price_prediction-using-LSTM
Prediction of Google Stock price using LSTM Model of Keras 

---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="1" id="C8X6lrZvjO3P"}
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
```
:::

::: {.cell .code execution_count="2" id="CEo3Af3EjYnd"}
``` python
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
```
:::

::: {.cell .code execution_count="3" id="XBbTuvPikMHg"}
``` python
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```
:::

::: {.cell .code execution_count="4" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="fBqajd_fkSdW" outputId="f8f87452-56fe-42b9-a4ee-1f4bfd49bc85"}
``` python
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

print(regressor.summary())
```

::: {.output .stream .stdout}
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 60, 50)            10400     
                                                                     
     dropout (Dropout)           (None, 60, 50)            0         
                                                                     
     lstm_1 (LSTM)               (None, 60, 50)            20200     
                                                                     
     dropout_1 (Dropout)         (None, 60, 50)            0         
                                                                     
     lstm_2 (LSTM)               (None, 60, 50)            20200     
                                                                     
     dropout_2 (Dropout)         (None, 60, 50)            0         
                                                                     
     lstm_3 (LSTM)               (None, 50)                20200     
                                                                     
     dropout_3 (Dropout)         (None, 50)                0         
                                                                     
     dense (Dense)               (None, 1)                 51        
                                                                     
    =================================================================
    Total params: 71,051
    Trainable params: 71,051
    Non-trainable params: 0
    _________________________________________________________________
    None
:::
:::

::: {.cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="OUFw4CL8kUXG" outputId="9f4531e1-e35b-4867-f60a-b2e760ba52cd"}
``` python
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```

::: {.output .stream .stdout}
    Epoch 1/100
    38/38 [==============================] - 14s 128ms/step - loss: 0.0422
    Epoch 2/100
    38/38 [==============================] - 6s 149ms/step - loss: 0.0067
    Epoch 3/100
    38/38 [==============================] - 5s 137ms/step - loss: 0.0053
    Epoch 4/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0054
    Epoch 5/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0056
    Epoch 6/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0052
    Epoch 7/100
    38/38 [==============================] - 6s 162ms/step - loss: 0.0049
    Epoch 8/100
    38/38 [==============================] - 5s 129ms/step - loss: 0.0044
    Epoch 9/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0041
    Epoch 10/100
    38/38 [==============================] - 6s 158ms/step - loss: 0.0046
    Epoch 11/100
    38/38 [==============================] - 5s 125ms/step - loss: 0.0043
    Epoch 12/100
    38/38 [==============================] - 6s 161ms/step - loss: 0.0042
    Epoch 13/100
    38/38 [==============================] - 5s 129ms/step - loss: 0.0037
    Epoch 14/100
    38/38 [==============================] - 5s 145ms/step - loss: 0.0040
    Epoch 15/100
    38/38 [==============================] - 6s 148ms/step - loss: 0.0038
    Epoch 16/100
    38/38 [==============================] - 5s 131ms/step - loss: 0.0035
    Epoch 17/100
    38/38 [==============================] - 6s 164ms/step - loss: 0.0037
    Epoch 18/100
    38/38 [==============================] - 5s 131ms/step - loss: 0.0042
    Epoch 19/100
    38/38 [==============================] - 6s 163ms/step - loss: 0.0032
    Epoch 20/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0037
    Epoch 21/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0034
    Epoch 22/100
    38/38 [==============================] - 6s 158ms/step - loss: 0.0036
    Epoch 23/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0031
    Epoch 24/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0037
    Epoch 25/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0031
    Epoch 26/100
    38/38 [==============================] - 5s 137ms/step - loss: 0.0033
    Epoch 27/100
    38/38 [==============================] - 6s 147ms/step - loss: 0.0030
    Epoch 28/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0033
    Epoch 29/100
    38/38 [==============================] - 6s 161ms/step - loss: 0.0030
    Epoch 30/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0036
    Epoch 31/100
    38/38 [==============================] - 6s 151ms/step - loss: 0.0028
    Epoch 32/100
    38/38 [==============================] - 5s 134ms/step - loss: 0.0030
    Epoch 33/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0030
    Epoch 34/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0029
    Epoch 35/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0028
    Epoch 36/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0025
    Epoch 37/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0024
    Epoch 38/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0025
    Epoch 39/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0030
    Epoch 40/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0025
    Epoch 41/100
    38/38 [==============================] - 6s 161ms/step - loss: 0.0026
    Epoch 42/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0025
    Epoch 43/100
    38/38 [==============================] - 5s 135ms/step - loss: 0.0025
    Epoch 44/100
    38/38 [==============================] - 6s 150ms/step - loss: 0.0027
    Epoch 45/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0023
    Epoch 46/100
    38/38 [==============================] - 6s 162ms/step - loss: 0.0025
    Epoch 47/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0026
    Epoch 48/100
    38/38 [==============================] - 6s 152ms/step - loss: 0.0023
    Epoch 49/100
    38/38 [==============================] - 5s 135ms/step - loss: 0.0024
    Epoch 50/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0026
    Epoch 51/100
    38/38 [==============================] - 6s 159ms/step - loss: 0.0021
    Epoch 52/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0024
    Epoch 53/100
    38/38 [==============================] - 6s 161ms/step - loss: 0.0025
    Epoch 54/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0023
    Epoch 55/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0022
    Epoch 56/100
    38/38 [==============================] - 6s 158ms/step - loss: 0.0023
    Epoch 57/100
    38/38 [==============================] - 5s 125ms/step - loss: 0.0022
    Epoch 58/100
    38/38 [==============================] - 6s 162ms/step - loss: 0.0022
    Epoch 59/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0018
    Epoch 60/100
    38/38 [==============================] - 5s 138ms/step - loss: 0.0020
    Epoch 61/100
    38/38 [==============================] - 6s 146ms/step - loss: 0.0021
    Epoch 62/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0019
    Epoch 63/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0020
    Epoch 64/100
    38/38 [==============================] - 5s 125ms/step - loss: 0.0021
    Epoch 65/100
    38/38 [==============================] - 6s 147ms/step - loss: 0.0021
    Epoch 66/100
    38/38 [==============================] - 5s 135ms/step - loss: 0.0019
    Epoch 67/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0019
    Epoch 68/100
    38/38 [==============================] - 6s 159ms/step - loss: 0.0019
    Epoch 69/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0017
    Epoch 70/100
    38/38 [==============================] - 6s 159ms/step - loss: 0.0017
    Epoch 71/100
    38/38 [==============================] - 5s 125ms/step - loss: 0.0019
    Epoch 72/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0020
    Epoch 73/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0020
    Epoch 74/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0017
    Epoch 75/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0017
    Epoch 76/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0017
    Epoch 77/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0019
    Epoch 78/100
    38/38 [==============================] - 6s 154ms/step - loss: 0.0017
    Epoch 79/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0016
    Epoch 80/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0016
    Epoch 81/100
    38/38 [==============================] - 5s 126ms/step - loss: 0.0015
    Epoch 82/100
    38/38 [==============================] - 5s 138ms/step - loss: 0.0016
    Epoch 83/100
    38/38 [==============================] - 6s 144ms/step - loss: 0.0016
    Epoch 84/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0015
    Epoch 85/100
    38/38 [==============================] - 6s 162ms/step - loss: 0.0015
    Epoch 86/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0016
    Epoch 87/100
    38/38 [==============================] - 6s 154ms/step - loss: 0.0016
    Epoch 88/100
    38/38 [==============================] - 5s 132ms/step - loss: 0.0016
    Epoch 89/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0016
    Epoch 90/100
    38/38 [==============================] - 6s 160ms/step - loss: 0.0016
    Epoch 91/100
    38/38 [==============================] - 5s 127ms/step - loss: 0.0015
    Epoch 92/100
    38/38 [==============================] - 6s 161ms/step - loss: 0.0016
    Epoch 93/100
    38/38 [==============================] - 5s 129ms/step - loss: 0.0016
    Epoch 94/100
    38/38 [==============================] - 5s 135ms/step - loss: 0.0013
    Epoch 95/100
    38/38 [==============================] - 6s 154ms/step - loss: 0.0014
    Epoch 96/100
    38/38 [==============================] - 5s 130ms/step - loss: 0.0015
    Epoch 97/100
    38/38 [==============================] - 6s 163ms/step - loss: 0.0015
    Epoch 98/100
    38/38 [==============================] - 5s 128ms/step - loss: 0.0015
    Epoch 99/100
    38/38 [==============================] - 6s 150ms/step - loss: 0.0018
    Epoch 100/100
    38/38 [==============================] - 5s 133ms/step - loss: 0.0015
:::

::: {.output .execute_result execution_count="5"}
    <keras.callbacks.History at 0x7ff988308e80>
:::
:::

::: {.cell .code execution_count="6" id="vqpWk988m8-c"}
``` python
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
```
:::

::: {.cell .code execution_count="7" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2gBfMQC5nBXU" outputId="9f5ffcd2-9865-4ce1-eca5-fe2c5f3d1d21"}
``` python
# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

::: {.output .stream .stdout}
    1/1 [==============================] - 2s 2s/step
:::
:::

::: {.cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":472}" id="BzRlT1WsnISk" outputId="e9dc5c40-21d3-4e9e-f3c9-01532da81fcf"}
``` python
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_b96ed50ba2da4880a757768e90cc0f9f/18cd016b9c21b2cbb59337531e343027a2fddab5.png)
:::
:::

::: {.cell .code execution_count="10" id="_L1IZ434n2qi"}
``` python
```
:::

