import cv2
import pytesseract
import numpy as np

# Set the path for Tesseract OCR (Update to your installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Paths for YOLO weights and configuration files
YOLO_WEIGHTS = r"C:\Users\user\Desktop\Edgematrix\day 5\yolov4.weights"
YOLO_CFG = r"C:\Users\user\Desktop\Edgematrix\day 5\yolov4.cfg"         # Configuration file
YOLO_CLASSES = r"C:\Users\user\Desktop\Edgematrix\day 5\coco.names"   # Classes file (download from YOLO repo)

# Load YOLO network
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open(YOLO_CLASSES, "r") as f:
    classes = f.read().strip().split("\n")

# Load input image
image_path = r"C:\Users\user\Desktop\Edgematrix\day 5\car5.png"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found. Check the path and file name.")

height, width = image.shape[:2]

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Run forward pass and collect detections
detections = net.forward(output_layers)

# Initialize lists to hold results
boxes, confidences, class_ids = [], [], []

# Process YOLO detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = int(np.argmax(scores))
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop through detections
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    # Draw rectangle and label on image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Detect and process number plates using OCR if the label is 'car'
    if label == "car":
        plate_image = image[y:y + h, x:x + w]
        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        gray_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)  # Reduce noise
        _, thresh_plate = cv2.threshold(gray_plate, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_plate = cv2.morphologyEx(thresh_plate, cv2.MORPH_CLOSE, kernel)
        
        # Contour detection to isolate plate area
        contours, _ = cv2.findContours(morph_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = w_c / h_c
            if 2 <= aspect_ratio <= 5:  # Typical license plate aspect ratio
                plate_roi = plate_image[y_c:y_c + h_c, x_c:x_c + w_c]
                resized_plate = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                
                # Tesseract OCR with custom configurations
                custom_config = '--oem 3 --psm 7'
                plate_text = pytesseract.image_to_string(resized_plate, config=custom_config)
                print(f"Detected Plate Text: {plate_text.strip()}")
                
                # Display detected text on the image
                cv2.putText(image, plate_text.strip(), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display results
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
