# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
Program to implement the linear regression using gradient descent.

Developed by: MALENI M

RegisterNumber: 212223040110 
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros (X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X = (data.iloc[1:, :-2].values) 
print (X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print('Name:MALENI M'    )
print('Register No.:212223040110'    )
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression (X1_Scaled, Y1_Scaled)
new_data = np.array ([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="1267" height="223" alt="Screenshot 2025-08-31 130114" src="https://github.com/user-attachments/assets/78e88632-5878-47c2-b2c6-fd3e8b687794" />
<img width="725" height="740" alt="Screenshot 2025-08-31 130151" src="https://github.com/user-attachments/assets/4bfb3b35-cd86-43db-9c13-1d11eef24d40" />
<img width="227" height="376" alt="Screenshot 2025-08-31 130212" src="https://github.com/user-attachments/assets/a9137775-f7a2-4ef4-ad3f-f70a578001ec" />
<img width="749" height="588" alt="Screenshot 2025-08-31 133925" src="https://github.com/user-attachments/assets/a356d2a3-378b-4faf-86c2-ee25feea40d7" />
<img width="203" height="467" alt="Screenshot 2025-08-31 130323" src="https://github.com/user-attachments/assets/1c0262bd-219e-4efe-8034-5dc747a7d3bc" />
<img width="404" height="37" alt="Screenshot 2025-08-31 130338" src="https://github.com/user-attachments/assets/d168bb86-6563-4104-901a-45c30e42ad6f" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
