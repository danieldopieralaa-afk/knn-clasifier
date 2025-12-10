import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
class KNN:
    def __init__(self,k):
        self.k=k
    def fit(self,X,Y):
        self.X_train=X
        self.Y_train=Y
    def predict(self, X):
        prediction=[]   

        for x in X: # x to cecha, ktora moze byc np. [0,3,5] gdzie kazdy element z listy okresla inna ceche np. dlugosc,szerokosc,pozycje
            distance = np.sum((self.X_train - x)**2, axis=1)
            k_idx=distance.argsort()[:self.k]
            k_labels=self.Y_train[k_idx]
            most_freq=Counter(k_labels).most_common(1)[0][0]
            prediction.append(most_freq)

        return np.array(prediction)


df = pd.read_csv("points.csv")
X=df[["x","y"]].values
Y=df["label"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model=KNN(k=1)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
accuracy = np.mean(Y_pred == Y_test)
full_pred = model.predict(X)
errors = full_pred != Y
correct = full_pred == Y

X_plot=df["x"]
Y_plot=df["y"]
C=df["label"]
x = np.linspace(-11, 11, 400) 
y = x                           
plt.figure(figsize=(8, 8))  
plt.xlim(-10.5, 10.5)
plt.ylim(-10.5, 10.5)
plt.scatter(
    X_plot[correct], Y_plot[correct],
    c=df["label"][correct],
    alpha=0.8,
    label="Poprawne"
)
plt.scatter(
    X_plot[errors], Y_plot[errors],
    c=df["label"][errors],  
    edgecolors='black',       
    linewidths=2,
    s=150,                     
    label="Błędne"
)

plt.axhline(0, color="black", linewidth=0.5)  
plt.axvline(0, color="black", linewidth=0.5)  
plt.plot(x,y,color="black", linewidth=0.5)
plt.plot(-x,y,color="black", linewidth=0.5)
plt.title("Punkty z pliku points.csv")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
print("Dokładność:", accuracy*100,"%")

    
