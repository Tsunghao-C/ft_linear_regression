import os


def predict_price() -> float:
    if os.path.exists("constants.txt"):
        with open("constants.txt", 'r') as f:
            constants = [float(line.strip()) for line in f]
    else:
        constants = [0, 0]
    theta_0, theta_1 = constants[0], constants[1]
    print("Enter an milage:", end=" ")
    mileage = float(input())
    return theta_0 + mileage * theta_1


def main():
    estimate_price = predict_price()
    print(f"Estimated price: {estimate_price}")


if __name__ == "__main__":
    main()
