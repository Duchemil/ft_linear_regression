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