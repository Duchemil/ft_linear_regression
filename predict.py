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
        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = predict_price(mileage, theta0, theta1)
        print(f"The estimated price for a car with {mileage} km is: {estimated_price:.2f}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")

# import pandas as pd

# def estimate_price(mileage, theta0, theta1):
#     """Calculate the estimated price using the linear regression model."""
#     return theta0 + (theta1 * mileage)

# def gradient_descent(x, y, theta0, theta1, learning_rate, iterations):
#     """Perform gradient descent to optimize theta0 and theta1."""
#     m = len(x)  # Number of data points
#     for _ in range(iterations):
#         # Calculate the temporary values for theta0 and theta1
#         tmp_theta0 = learning_rate * (1 / m) * sum(estimate_price(x[i], theta0, theta1) - y[i] for i in range(m))
#         tmp_theta1 = learning_rate * (1 / m) * sum((estimate_price(x[i], theta0, theta1) - y[i]) * x[i] for i in range(m))
        
#         # Simultaneously update theta0 and theta1
#         theta0 -= tmp_theta0
#         theta1 -= tmp_theta1
    
#     return theta0, theta1

# if __name__ == "__main__":
#     # Load the dataset
#     data = pd.read_csv('data.csv')
#     x = data['km'].tolist()
#     y = data['price'].tolist()

#     # Initialize parameters
#     theta0 = 0.0
#     theta1 = 0.0
#     learning_rate = 0.01  # Small step size
#     iterations = 1000  # Number of iterations

#     # Perform gradient descent
#     theta0, theta1 = gradient_descent(x, y, theta0, theta1, learning_rate, iterations)

#     print(f"Optimized Theta0: {theta0}")
#     print(f"Optimized Theta1: {theta1}")