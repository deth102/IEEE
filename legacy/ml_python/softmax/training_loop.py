from Model import SoftmaxLogisticTrainer
from prepare_data import X_train, y_train, X_val, y_val, X_test, y_test
import numpy as np
import pickle
trainer = SoftmaxLogisticTrainer(
    lr=2.2,
    l2=1e-4,
    epochs=500
)

model = trainer.fit(
    X_train, y_train,
    X_val, y_val,
    patience=20
)

with open("softmax_logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
    

