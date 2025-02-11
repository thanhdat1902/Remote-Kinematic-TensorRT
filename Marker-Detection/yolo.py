from ultralytics import YOLO
# Display model information (optional)
if __name__ == '__main__':
    # Build a YOLOv9c model from scratch
    model = YOLO("yolov9e.yaml")


    # Build a YOLOv9c model from pretrained weight
    model = YOLO("yolov9e.pt")
    model.info()

    epochs = 100  # Initial number of epochs
    batch_size = 8  # Adjust based on your GPU memory
    patience = 10  # Early stopping patience

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="./data_mobile.yaml", epochs=epochs, imgsz=640, batch=batch_size, patience=patience)

    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    results = model("image.png")
