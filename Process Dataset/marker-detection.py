from ultralytics import YOLO
# Display model information (optional)
if __name__ == '__main__':
    model = YOLO("models/best-marker.pt") 

    # Train the model
    results = model.predict("./frame_003634 (1).PNG", save=True)
