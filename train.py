import pandas as pd
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('data.csv')
    x = data['km'].tolist()
    y = data['price'].tolist()
except Exception as e:
    print("\033[H\033[J", end="")
    print("An error occurred while processing the data.")


n = len(x)
if n > 0:
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Calculate the slope (thet1)
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    theta1 = numerator / denominator

    # Calculate the intercept (theta0)
    theta0 = mean_y - theta1 * mean_x

    with open('model.txt', 'w') as file:
        file.write(f"theta0: {theta0}\n")
        file.write(f"theta1: {theta1}\n")

    # Clear the console
    print("\033[H\033[J", end="")
    print(f"Theta0 (Intercept): {theta0}, Theta1 (Slope): {theta1}")

    plt.scatter(x, y)
    regression_line = [theta0 + theta1 * i for i in x]
    plt.plot(x, regression_line, color='red', label='Regression Line')
    plt.xlabel('Km')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Linear Regression')
    plt.show()
else:
    # Clear the console and print no data points
    print("\033[H\033[J", end="")
    print("No data points available for regression.")