from ultralytics import YOLO
# Display model information (optional)
if __name__ == '__main__':
    model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML
    model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

    epochs = 120  # Initial number of epochs
    batch_size = 8  # Adjust based on your GPU memory
    patience = 20  # Early stopping patience

    # Train the model
    results = model.train(data="./data_mobile.yaml", epochs=epochs, imgsz=640, batch=batch_size, patience=patience, pose=8)


