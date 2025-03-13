import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ft_LinearRegressionGD:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_0 = 0
        self.theta_1 = 0
    
    def fit(self, X, y):
        m = len(X)

        for _ in range(self.epochs):
            gradient_theta_0 = -2 * np.sum(y - self.theta_1 * X + self.theta_0) / m
            gradient_theta_1 = -2 * np.sum(X * (y - (self.theta_1 * X + self.theta_0))) / m

            self.theta_0 = self.theta_0 + (self.learning_rate * gradient_theta_0)
            self.theta_1 = self.theta_1 - (self.learning_rate * gradient_theta_1)
            # y_pred = self.theta_0 + self.theta_1 * X
            # error = y_pred - y

            # # Compute gradients
            # d_theta_0 = np.sum(error) / m
            # d_theta_1 = np.sum(error * X) / m
            # print(d_theta_0, d_theta_1)

            # # update parameters
            # self.theta_0 -= self.learning_rate * d_theta_0
            # self.theta_1 -= self.learning_rate * d_theta_1
        # mean_x, mean_y = sum(X) / n, sum(y) / n

        # numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        # denominator = sum((X[i] - mean_x) ** 2 for i in range(n))

        # self.theta_1 = numerator / denominator
        # self.theta_0 = mean_y - self.theta_1 * mean_x

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
    with open("constants.txt", 'w') as f:
        f.write("\n".join([str(lr.theta_0), str(lr.theta_1)]))

    # bonus 1 & 2 plot data and linear regression line
    plots(lr, X, y)
    # bonus 3 evaluation
    evaluation(lr, X, y)


if __name__ == "__main__":
    main()
