# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 00:10:25 2020

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_regression.csv", sep =";")

y = df.araba_max_hiz.values.reshape(-1,1) #values arraye dönüştürür.reshape skylearn kütüphaeisnde kullanmak için yapmamız gerek.
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba max hızı")
plt.ylabel("araba fiyatı")
plt.show()

#linear regression  y=b0+b1*x
#multiple linear regression y=b0+b1*x1+b2*x2

# %% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y) # x ve y datamıza en uygun line yı fit ediyoruz. mse en uygun olcak şelilde.


#lr.predict(x) # elde ettiğiim line ya göre her bir değere göre tahmin.

#%% predict
y_head = lr.predict(x)
plt.plot(x,y_head, color="red")
plt.show()
