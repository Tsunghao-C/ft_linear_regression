import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ft_LinearRegression:
    def __init__(self):
        self.theta_0 = 0
        self.theta_1 = 0
    
    def fit(self, X, y):
        n = len(X)
        mean_x, mean_y = sum(X) / n, sum(y) / n

        numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((X[i] - mean_x) ** 2 for i in range(n))

        self.theta_1 = numerator / denominator
        self.theta_0 = mean_y - self.theta_1 * mean_x

    def predict(self, milage) -> float:
        if isinstance(milage, float | int):
            return self.theta_1 * milage + self.theta_0
        else:
            return [self.theta_1 * x + self.theta_0 for x in milage]


def plots(lr: ft_LinearRegression, X, y):
    lr_x = [x for x in range(30000, 250000, 1000)]
    lr_y = lr.predict(lr_x)
    plt.scatter(X, y)
    plt.plot(lr_x, lr_y, color="red")
    plt.text(130000, 6500, f'y = {lr.theta_0:.4f} {lr.theta_1:.4f} * x', color="red")
    plt.xlabel("Milage")
    plt.ylabel("Price")
    plt.title("Milage - Price scatter plot")
    plt.show()


def evaluation(lr: ft_LinearRegression, X, y):
    lr_y = lr.predict(X)
    mse = sum([(y1 - y2) ** 2 for y1, y2 in zip(y, lr_y)]) / len(lr_y)
    rmse = mse ** 0.5
    mae = sum([abs(y1 - y2) for y1, y2 in zip(y, lr_y)]) / len(lr_y)
    y_mean = sum(y) / len(y)
    SS_total = sum([(y1 - y_mean) ** 2 for y1 in y])
    SS_pred = sum([(y1 - y2) ** 2 for y1, y2 in zip(y, lr_y)])
    rsquare = 1 - (SS_pred / SS_total)  # Coefficient of determinant
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2-score: {rsquare:.4f}")


def main():
    # mandatory 1: receive input and generate estimate
    print("Enter a milage:", end=" ")
    mile = float(input())
    lr = ft_LinearRegression()
    est_price = lr.predict(mile)
    print(f"Estimated price before training: {est_price}")

    # mandatory 2: train model with data
    df = pd.read_csv("data.csv")
    X = df['km']
    y = df['price']
    lr.fit(X, y)
    print("Trained constants (theta_0, theta_1):", (lr.theta_0, lr.theta_1))
    print(f"Estimated price after trainig: {lr.predict(mile)}")

    # bonus 1 & 2 plot data and linear regression line
    plots(lr, X, y)
    # bonus 3 evaluation
    evaluation(lr, X, y)




if __name__ == "__main__":
    main()
