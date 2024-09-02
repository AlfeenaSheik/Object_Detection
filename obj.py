import cv2
import numpy as np
import winsound

# Define sound effects
sound_effects = {
    'car': 'car.wav',
    'cat':'cat.wav',
    'dog': 'dog.wav',
    'bird':'bird.wav',
    'truck':'truck.wav',
    'clock':'clock.wav',
    
}

educational_facts = {
    'car': 'A car is a vehicle that moves on four wheels.',
    'motorbike': 'A bike is a two-wheeled vehicle powered by pedaling.',
    'Scissors':"Scissors have been around for more than 3,000 years! The first ones were made by the Egyptians and were a bit tricky to use compared to today's scissors.",
    'clock':"Before clocks, people used the sun to tell time with something called a sundial. The shadow would move around as the sun moved across the sky!",
    'toothbrush':"The average person will spend 38.5 days brushing their teeth over a lifetime! That's more than a month just to keep your teeth clean."
}

def display_fact(frame, label, x, y):
    fact = educational_facts.get(label)
    cv2.putText(frame, fact, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 80), 2)

def play_sound(label):
    if label in sound_effects:
        winsound.PlaySound(sound_effects[label], winsound.SND_FILENAME | winsound.SND_ASYNC)

# Paths to YOLO model files
cfg_path = 'yolov3.cfg'  # Ensure this is the path to your .cfg file
weights_path = 'yolov3.weights'  # Ensure this is the path to your .weights file
names_path = 'coco.names'  # Ensure this is the path to your .names file

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()

# Fix: Adjust for scalar or list output
unconnected_out_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_out_layers, np.ndarray):
    unconnected_out_layers = unconnected_out_layers.flatten()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture for laptop camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

# Get screen size
screen_width = 1900 # Update this based on your screen resolution
screen_height = 1080  # Update this based on your screen resolution

# Set window size to 3/4th of the screen
window_width = int(screen_width * 0.75)
window_height = int(screen_height * 0.75)

# Create a named window with the desired size
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', window_width, window_height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame for the model
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detection results
    class_ids = []
    confidences = []
    boxes = []
    height, width, _ = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            play_sound(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         
            display_fact(frame, label, x, y)

            

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
