from ultralytics import YOLO
import os
import glob

# Load YOLOv8 model (ensure the path to the model is correct)
model = YOLO('yolov8n.pt')  # Change to a larger model if needed

# Hyperparameters for fine-tuning
learning_rate = 0.001  # You can try smaller values like 0.0005 or 0.0001
batch_size = 16  # Adjust based on your hardware, try 32 or 64 if possible
epochs = 50  # Increase epochs for more training time
imgsz = 640  # Image size (adjust for your dataset)

# Data Augmentation (If you use YOLOv8, this is handled automatically in the data.yaml file)
# Ensure your `data.yaml` file has proper augmentation settings

# Train the model
try:
    results = model.train(
        data='data.yaml',  # Path to your dataset configuration file
        epochs=epochs,  # Increase epochs
        imgsz=imgsz,  # Image size
        batch=batch_size,  # Batch size
        lr0=learning_rate,  # Initial learning rate
        lrf=0.1,  # Final learning rate (use a lower value to avoid overfitting)
        name='elephant_detection_model',  # Base name for model folder
        workers=4,  # Number of workers for loading the dataset
    )
    print("Training completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# Find the most recent training run folder
runs_dir = 'runs/detect/'
model_folders = glob.glob(os.path.join(runs_dir, 'elephant_detection_model*'))

if model_folders:
    # Sort the model folders based on the creation time (latest folder first)
    latest_model_folder = max(model_folders, key=os.path.getctime)
    results_path = os.path.join(latest_model_folder, 'results.csv')

    if os.path.exists(results_path):
        # Move results.csv to the main project directory
        try:
            os.rename(results_path, 'results.csv')
            print(f"Results moved to the project folder as results.csv.")
        except Exception as e:
            print(f"Error moving results file: {e}")
    else:
        print(f"results.csv not found in {latest_model_folder}. Ensure training completed successfully.")
else:
    print(f"No model folders found in {runs_dir}. Ensure training completed successfully.")
    
# Optional cleanup: Remove unnecessary lines if causing errors
# Uncomment this line if any problematic code is present
# del results  # Example cleanup if results is unnecessary










# from ultralytics import YOLO
# import pandas as pd
# import os
# import glob

# # Load YOLOv8 model
# model = YOLO('yolov8n.pt')  # Replace with the correct path if needed

# # Train the model
# results = model.train(
#     data='data.yaml',
#     epochs=3,  # Adjust based on your requirements
#     imgsz=640,
#     name='elephant_detection_model',  # Base name for model folder
# )

# # Find the most recent training run folder
# runs_dir = 'runs/detect/'
# model_folders = glob.glob(os.path.join(runs_dir, 'elephant_detection_model*'))
# if model_folders:
#     # Sort the model folders based on the creation time (latest folder first)
#     latest_model_folder = max(model_folders, key=os.path.getctime)
#     results_path = os.path.join(latest_model_folder, 'results.csv')

#     if os.path.exists(results_path):
#         # Move results.csv to the main project directory
#         os.rename(results_path, 'results.csv')
#         print(f"Results moved to the project folder as results.csv.")
#     else:
#         print(f"results.csv not found in {latest_model_folder}. Ensure training completed successfully.")
# else:
#     print(f"No model folders found in {runs_dir}. Ensure training completed successfully.")

# # Optional cleanup: Remove unnecessary lines if causing errors
# # Uncomment this line if any problematic code is present
# # del results  # Example cleanup if results is unnecessary
