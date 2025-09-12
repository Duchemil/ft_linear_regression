import pandas as pd
import matplotlib.pyplot as plt

def calculate_errors(y_true, y_pred):
    """Calculate Mean Average Error."""
    n = len(y_true)
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    return mae

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

        mae= calculate_errors(y, [theta0 + theta1 * i for i in x])
        print("Precision : ")
        print(f"Mean Absolute Error: {mae}")
        
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

def plot_training_evolution(history_csv='training_history.csv', data_csv='data.csv'):
    """Plot multiple regression lines from training_history.csv over the dataset."""
    try:
        data = pd.read_csv(data_csv)
        x = data['km'].tolist()
        y = data['price'].tolist()
        hist = pd.read_csv(history_csv)
    except Exception as e:
        print("Could not load data or training history:", e)
        return

    # Scatter original data
    plt.scatter(x, y, s=12, label='Data')

    # Plot one line per saved (theta0, theta1)
    n_hist = len(hist)
    for idx, row in hist.iterrows():
        t0 = float(row['theta0'])
        t1 = float(row['theta1'])
        y_line = [t0 + t1 * xi for xi in x]

        # Styling: Change colors and labels based on iteration milestones
        iteration = int(row['iteration'])
        if iteration == 1:
            color = 'blue'
            label = "1st iteration"
        elif iteration == 10:
            color = 'orange'
            label = "10th iteration"
        elif iteration == 100:
            color = 'purple'
            label = "100th iteration"
        elif idx == n_hist - 1:  # Final iteration
            color = 'green'
            label = "Final iteration (1000th)"
        else:
            color = 'red'
            label = None

        alpha = max(0.15, (idx + 1) / n_hist)
        lw = 2.5 if idx == n_hist - 1 else 1.0

        plt.plot(x, y_line, color=color, alpha=alpha, linewidth=lw, label=label)

    plt.xlabel('Km')
    plt.ylabel('Price')
    plt.title('Gradient Descent Evolution (from training_history.csv)')
    plt.legend()
    plt.tight_layout()
    plt.show()

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

        # Plot GD evolution from CSV
        plot_training_evolution('training_history.csv', 'data.csv')
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
