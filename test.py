"""
99.8% on mnist
"""

import numpy as np
from nn import *
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist 



# ------------------- LOAD DATA -------------------

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0
training_data = [
    (train_X[i].reshape(1, 784), np.eye(10)[train_y[i]].reshape(1, 10))
    for i in range(len(train_X))
]
test_data = [
    (test_X[i].reshape(1, 784), test_y[i]) 
    for i in range(len(test_X))
]



# ------------------- CREATE NN -------------------

model = SequentialNN([
    Linear(28*28, 128),
    ReLU(),
    Linear(128, 128),
    ReLU(),
    Linear(128, 10),
    Softmax()
])



# ------------------- HYPERPARAMETERS -------------------

learning_rate = 0.6
epochs = 100
batch_size = 100



# ------------------- TRAIN -------------------

print("Training")
for epoch in range(epochs):
    epoch_loss = 0.0
    nb_reussi = 0
    for i in range( int(len(training_data) / batch_size) ):
        batch_loss = Tensor(0)

        for x_np, y_np in training_data[i*batch_size:(i+1)*batch_size]:
            x = Tensor(x_np)
            y_true = Tensor(y_np)
        
            # --- FORWARD PASS ---
            y_pred = model(x)
            if np.argmax(y_pred.data) == np.argmax(y_true.data):
                nb_reussi += 1
            
            # --- CALCUL DE LA PERTE (MSE) ---
            diff = y_pred - y_true
            loss = np.sum(diff * diff)  # On fait la somme sur les 10 classes
            batch_loss += loss
        
        epoch_loss += batch_loss.data
        # --- BACKWARD PASS ---
        model.zero_grad()
        batch_loss.backward()
        
        # --- OPTIMISATION (SGD) ---
        for p in model.parameters():
            p.data -= (learning_rate * p.grad) / batch_size

    if (epoch > 50):
        learning_rate = 0.33

    l = np.mean(epoch_loss / 60000)
    print(f"Epoch {epoch} | Loss: {l} | ratio: {nb_reussi / 60000}")
