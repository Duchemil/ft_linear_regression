import pandas as pd
import matplotlib.pyplot as plt

def plotting_data():
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


def load_model(file_path):
    """Load theta0 and theta1 from the model file."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            theta0 = float(lines[0].split(":")[1].strip())
            theta1 = float(lines[1].split(":")[1].strip())
        return theta0, theta1
    except FileNotFoundError:
        print("Model file not found. Using default values for theta0 and theta1.")
        return 0.0, 0.0
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return 0.0, 0.0


def predict_price(mileage, theta0, theta1):
    """Predict the price based on mileage using the linear regression model."""
    return theta0 + (theta1 * mileage)


if __name__ == "__main__":
    # Load the model parameters
    theta0, theta1 = load_model('model.txt')

    # Prompt the user for mileage
    try:
        print("\033[H\033[J", end="")
        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = predict_price(mileage, theta0, theta1)
        print(f"The estimated price for a car with {mileage} km is: {estimated_price:.2f}")
        plotting_data()
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
