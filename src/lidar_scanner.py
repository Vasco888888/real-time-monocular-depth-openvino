import cv2
import numpy as np
import openvino as ov
import os

# Setup OpenVINO
core = ov.Core()
model_xml = "models/MiDaS_small.xml"
if not os.path.exists(model_xml):
    print("Run download_model.py first.")
    exit()

model = core.read_model(model=model_xml)
compiled_model = core.compile_model(model=model, device_name="GPU")
output_layer = compiled_model.output(0)

cap = cv2.VideoCapture(0)

# Scanner and Smoothing Variables
smoothed_depth = None
alpha = 0.2     # Smoothing factor (0.1 = very smooth/slow, 0.9 = fast/flickery)
scan_val = 0
direction = 2   # Laser speed
tolerance = 5   # Laser thickness

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocessing
    img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = img.transpose(2, 0, 1)
    input_data = np.expand_dims(input_data, 0).astype(np.float32)
    
    # Inference and Stabilization
    result = compiled_model([input_data])[output_layer]
    depth_map = result[0] 
    
    if smoothed_depth is None:
        smoothed_depth = depth_map
    else:
        smoothed_depth = cv2.addWeighted(smoothed_depth, 1 - alpha, depth_map, alpha, 0)

    # Postprocessing
    depth_norm = cv2.normalize(smoothed_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    depth_resized = cv2.resize(depth_norm, (frame.shape[1], frame.shape[0]))

    # Volumetric Scan Logic
    scan_val += direction
    if scan_val >= 255 or scan_val <= 0:
        direction *= -1

    mask = cv2.inRange(depth_resized, scan_val - tolerance, scan_val + tolerance)

    # HUD
    display_frame = cv2.addWeighted(frame, 0.4, frame, 0, 0)
    display_frame[mask > 0] = (0, 255, 0) 
    cv2.putText(display_frame, f"LiDAR RANGE: {scan_val}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display
    cv2.imshow('Lidar Scanner', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()