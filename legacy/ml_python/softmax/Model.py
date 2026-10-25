import numpy as np

class SoftmaxLogisticModel:
    """
    Multiclass Logistic Regression (Softmax)
    - Inference
    - Feature normalization INSIDE model
    """

    def __init__(self, W, b, mean, std):
        self.W = W          # (C, D) C: num classes, D: num features
        self.b = b          # (C,)
        self.mean = mean    # (D,)
        self.std = std      # (D,)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def linear(self, x):
        x = self.normalize(x)
        return self.W @ x + self.b

    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def predict_proba(self, X):
        """
        X: (n_samples, n_features)
        return: (n_samples, n_classes)
        """
        X = np.asarray(X)

        
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Z = np.array([self.linear(x) for x in X])   # (N, C)
        Z -= np.max(Z, axis=1, keepdims=True)       # numerical stability

        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


    def predict(self, X):
        """
        X: (n_samples, n_features)
        return: (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def contribution(self, x):
        """
        Feature contribution per class
        return shape: (C, D)
        """
        x_norm = self.normalize(x)
        return self.W * x_norm
class SoftmaxLogisticTrainer:
    def __init__(self, lr, l2, epochs):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def softmax_loss(Xn, y, W, b):
        Z = Xn @ W.T + b # Xn: (N, D), W: (C, D), b: (C,) => Z: (N, C)
        Z -= np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-9))

    def fit(self, X, y, X_val=None, y_val=None, patience=None):
        """
        X, y       : train set
        X_val,y_val: validation set
        patience   : None -> không early stopping
                    int  -> early stopping theo val loss
        """

        N, D = X.shape
        C = np.max(y) + 1

        # ===== FIT NORMALIZATION (TRAIN ONLY) =====
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8

        Xn = (X - mean) / std
        if X_val is not None:
            Xv = (X_val - mean) / std

        # ===== INIT =====
        W = np.zeros((C, D))
        b = np.zeros(C)

        Y = np.zeros((N, C))
        Y[np.arange(N), y] = 1.0

        # ===== LOG STORAGE =====
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []

        # ===== EARLY STOPPING STATE =====
        best_val_loss = np.inf
        best_W = None
        best_b = None
        wait = 0

        for epoch in range(self.epochs):

            # ===== FORWARD (TRAIN) =====
            Z = Xn @ W.T + b
            Z -= np.max(Z, axis=1, keepdims=True)

            exp_Z = np.exp(Z)
            P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

            # ===== TRAIN LOSS =====
            train_loss = -np.mean(np.log(P[np.arange(N), y] + 1e-9))
            self.train_losses.append(train_loss)

            # ===== TRAIN ACC =====
            y_pred_train = np.argmax(P, axis=1)
            train_acc = np.mean(y_pred_train == y)
            self.train_accs.append(train_acc)

            # ===== BACKWARD =====
            diff = P - Y
            dW = diff.T @ Xn / N + 2 * self.l2 * W
            db = diff.mean(axis=0)

            # ===== UPDATE =====
            W -= self.lr * dW
            b -= self.lr * db

            # ===== VALIDATION =====
            if X_val is not None:
                Zv = Xv @ W.T + b
                Zv -= np.max(Zv, axis=1, keepdims=True)

                exp_Zv = np.exp(Zv)
                Pv = exp_Zv / np.sum(exp_Zv, axis=1, keepdims=True)

                val_loss = -np.mean(np.log(Pv[np.arange(len(y_val)), y_val] + 1e-9))
                self.val_losses.append(val_loss)

                y_pred_val = np.argmax(Pv, axis=1)
                val_acc = np.mean(y_pred_val == y_val)
                self.val_accs.append(val_acc)

                # ===== PRINT =====
                print(
                    f"Epoch {epoch+1:03d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

                # ===== EARLY STOPPING =====
                if patience is not None:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_W = W.copy()
                        best_b = b.copy()
                        wait = 0
                    else:
                        wait += 1

                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        W, b = best_W, best_b
                        break
            else:
                print(
                    f"Epoch {epoch+1:03d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Train Acc: {train_acc:.4f}"
                )

        # ===== RETURN MODEL (WITH NORMALIZATION) =====
        return SoftmaxLogisticModel(W, b, mean, std)


