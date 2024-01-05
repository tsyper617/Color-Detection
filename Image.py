import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the color dataset
color_data = pd.read_csv('finalcolors.csv')
color_data.rename(columns={'label': 'color'}, inplace=True)

# Features (RGB values)
X = color_data[['red', 'green', 'blue']]

# Target variable (Color names)
y = color_data['color']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Function to predict color based on RGB values
def predict_color(red, green, blue):
    # Predict the color using the trained model
    predicted_color = model.predict([[red, green, blue]])
    return predicted_color[0]

# Callback function for mouse events
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)
        predicted_color_name = predict_color(r, g, b)
        colors.append((r, g, b, predicted_color_name))  # Append predicted color to the list

        print(predicted_color_name)
        # Log outputs to a file
        with open('output_log.txt', 'a') as f:
            f.write(f"RGB: ({r}, {g}, {b}), Predicted Color: {predicted_color_name}\n")

# Manually specify the image path
img_path = 'mm.jpeg'  # Replace this with your image path

# Read the image using OpenCV
img = cv2.imread(img_path)

# Create a named window
cv2.namedWindow('image')

# Set the mouse callback
cv2.setMouseCallback('image', draw_function)

# Initialize an empty list to store colors
colors = []

# Display the image
cv2.imshow('image', img)

# Keep the window open until a key is pressed
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()

# Print a message to indicate the process has finished
print("Processing complete. Please check 'output_log.txt' for the recorded outputs.")
