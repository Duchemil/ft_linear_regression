import pandas as pd
import matplotlib.pyplot as plt
import csv

result_training = []

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

def estimate_price(mileage, theta0, theta1):
    """Calculate the estimated price using the linear regression model."""
    return theta0 + (theta1 * mileage)

def gradient_descent(x, y, theta0, theta1, learning_rate, iterations):
    """Perform gradient descent to optimize theta0 and theta1."""
    m = len(x)  # Number of data points
    
    x_max = max(x)
    y_max = max(y)
    x_norm = [xi / x_max for xi in x]
    y_norm = [yi / y_max for yi in y]
    
    for i in range(1, iterations + 1):
        # Calculate the temporary values for theta0 and theta1
        tmp_theta0 = learning_rate * (1 / m) * sum(estimate_price(x_norm[j], theta0, theta1) - y_norm[j] for j in range(m))
        tmp_theta1 = learning_rate * (1 / m) * sum((estimate_price(x_norm[j], theta0, theta1) - y_norm[j]) * x_norm[j] for j in range(m))
        
        # Simultaneously update theta0 and theta1
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        
        # Record theta values at specified steps
        if i in {1, 10, 100, 1000}:
            result_training.append((i, theta0, theta1))
    
    return theta0, theta1, result_training

def save_model(file_path, theta0, theta1):
    with open(file_path, 'w') as f:
        f.write(f"theta0: {theta0}\n")
        f.write(f"theta1: {theta1}\n")

def save_history_csv(file_path, history_norm, x_max, y_max):
    """Save (iteration, theta0, theta1) in original scale for plotting."""
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'theta0', 'theta1'])
        for it, t0n, t1n in history_norm:
            t0 = y_max * t0n
            t1 = (y_max * t1n) / x_max
            writer.writerow([it, t0, t1])

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('data.csv')
    x = data['km'].tolist()
    y = data['price'].tolist()

    # Initialize parameters
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.1  # Small step size
    iterations = 1000  # Number of iterations

    # Train
    theta0_norm, theta1_norm, history_norm = gradient_descent(x, y, theta0, theta1, learning_rate, iterations)

    # Convert final thetas to original scale and save the model
    x_max, y_max = max(x), max(y)
    theta0_final = y_max * theta0_norm
    theta1_final = (y_max * theta1_norm) / x_max
    print(f"Optimized Theta0 (Intercept): {theta0_final}, Theta1 (Slope): {theta1_final}")
    save_model('model.txt', theta0_final, theta1_final)

    # Save training history (original scale) for plotting in predict.py
    save_history_csv('training_history.csv', history_norm, x_max, y_max)