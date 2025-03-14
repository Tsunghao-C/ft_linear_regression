import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ft_LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=3000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_0 = 0
        self.theta_1 = 0
        self.x_mean = None
        self.x_std = None
        self.training_rounds = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        m = len(X)

        # Normalize X and store scaled factors in the object variable
        # If you don't normalized X, the number will be too big and unable to converge
        self.x_mean = np.mean(X)
        self.x_std = np.std(X)
        X_scaled = (X - self.x_mean) / self.x_std

        # Gradient descent interations
        for _ in range(self.epochs):
            pred = self.theta_0 + self.theta_1 * X_scaled
            error = y - pred
            # Loss function (MSE) = np.sum(error ** 2) / m
            # calculate the gradient of Loss function partial derivitate to "theta_0" and "theta_1" respectively
            gradient_theta_0 = -2 * np.sum(error) / m
            gradient_theta_1 = -2 * np.sum(X_scaled * error) / m

            print("gradient", gradient_theta_0, gradient_theta_1)

            step_theta_0 = gradient_theta_0 * self.learning_rate
            step_theta_1 = gradient_theta_1 * self.learning_rate

            print("step", step_theta_0, step_theta_1)

            # Update the thetas
            self.theta_0 -= step_theta_0
            self.theta_1 -= step_theta_1

            # Stop early if gradient is too small
            self.training_rounds += 1
            if abs(gradient_theta_0) < 1e-6 and abs(gradient_theta_1) < 1e-6:
                break
        
        # transform back to unscaled value
        self.theta_1 /= self.x_std
        self.theta_0 -= self.theta_1 * self.x_mean

    def predict(self, X) -> float:
        if isinstance(X, float | int):
            return self.theta_1 * X + self.theta_0
        else:
            return [self.theta_1 * i + self.theta_0 for i in X]


def plots(lr: ft_LinearRegressionGD, X, y):
    # create regression line data
    lr_x = [x for x in range(30000, 250000, 1000)]
    lr_y = lr.predict(lr_x)

    # Draw data points
    plt.scatter(X, y)
    # Draw regression line
    plt.plot(lr_x, lr_y, color="red")
    plt.text(130000, 6500, f'y = {lr.theta_0:.4f} {lr.theta_1:.4f} * x', color="red")
    plt.xlabel("Milage")
    plt.ylabel("Price")
    plt.title("Milage - Price scatter plot")
    plt.savefig("linear_regression.png")
    # plt.show()


def evaluation(lr: ft_LinearRegressionGD, X, y):
    lr_y = lr.predict(X)
    mse = sum([(y1 - y2) ** 2 for y1, y2 in zip(y, lr_y)]) / len(lr_y)
    rmse = mse ** 0.5
    mae = sum([abs(y1 - y2) for y1, y2 in zip(y, lr_y)]) / len(lr_y)

    # R2 score
    y_mean = sum(y) / len(y)
    SS_total = sum([(y1 - y_mean) ** 2 for y1 in y])
    SS_pred = sum([(y1 - y2) ** 2 for y1, y2 in zip(y, lr_y)])
    rsquare = 1 - (SS_pred / SS_total)  # Coefficient of determinant

    print("\nEvaluations for LR:\n")
    print(f"{'Metric':<15} {'Value':<15}")
    print("-" * 30)
    print(f"{'MSE':<15} {mse:<15.4f}")
    print(f"{'RMSE':<15} {rmse:<15.4f}")
    print(f"{'MAE':<15} {mae:<15.4f}")
    print(f"{'R2-score':<15} {rsquare:<15.4f}")


def main():
    # mandatory 2: train model with data
    df = pd.read_csv("data.csv")
    lr = ft_LinearRegressionGD()
    X = df['km']
    y = df['price']
    lr.fit(X, y)
    print("(theta_0, theta_1):", (lr.theta_0, lr.theta_1))
    print("Training rounds:", lr.training_rounds)
    with open("constants.txt", 'w') as f:
        f.write("\n".join([str(lr.theta_0), str(lr.theta_1)]))

    # bonus 1 & 2 plot data and linear regression line
    plots(lr, X, y)
    # bonus 3 evaluation
    evaluation(lr, X, y)


if __name__ == "__main__":
    main()
