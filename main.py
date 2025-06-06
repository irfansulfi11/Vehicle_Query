import cv2
import numpy as np
from ultralytics import YOLO
import argparse

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for lightweight model; replace with yolov8m.pt for better accuracy

# Define vehicle classes from COCO dataset
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
class_ids = [2, 5, 7, 3]  # COCO class IDs for car, truck, bus, motorcycle

# Define color ranges in HSV for detection (all as lists of tuples, adjusted for red)
color_ranges = {
    'red': [((0, 70, 50), (15, 255, 255)), ((160, 70, 50), (180, 255, 255))],  # Broadened red ranges
    'blue': [((100, 70, 50), (130, 255, 255))],
    'green': [((40, 70, 50), (80, 255, 255))],
    'white': [((0, 0, 180), (180, 30, 255))],
    'black': [((0, 0, 0), (180, 255, 30))],
    'yellow': [((20, 70, 100), (40, 255, 255))]  # Adjusted yellow range
}

def get_dominant_color(image, box, debug=False):
    """Extract dominant color from a bounding box region in HSV."""
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        if debug:
            print(f"Empty ROI for box {box}")
        return None
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_roi, axis=(0, 1))
    
    if debug:
        print(f"Box {box}, Mean HSV: {mean_hsv}")
    
    for color_name, ranges in color_ranges.items():
        for lower, upper in ranges:
            if (lower[0] <= mean_hsv[0] <= upper[0] and
                lower[1] <= mean_hsv[1] <= upper[1] and
                lower[2] <= mean_hsv[2] <= upper[2]):
                if debug:
                    print(f"Detected color: {color_name} for box {box}")
                return color_name
    if debug:
        print(f"No color match for box {box}, HSV: {mean_hsv}")
    return None

def detect_vehicles_in_video(video_path, confidence_threshold=0.5, debug=False):
    """Detect vehicles and their colors directly from video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    detections = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to temporary image for YOLO processing
        results = model(frame, conf=confidence_threshold)
        frame_detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids_detected = result.boxes.cls.cpu().numpy()
            
            for box, score, cls_id in zip(boxes, scores, class_ids_detected):
                if int(cls_id) in class_ids:
                    vehicle_type = vehicle_classes[class_ids.index(int(cls_id))]
                    color = get_dominant_color(frame, box, debug=debug)
                    if color:
                        frame_detections.append({
                            'vehicle_type': vehicle_type,
                            'color': color,
                            'box': box,
                            'confidence': score
                        })
                        if debug and color == 'red':
                            # Save frame with red vehicle for verification
                            output_frame = frame.copy()
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.imwrite(f"debug_red_{frame_idx}_{vehicle_type}.jpg", output_frame)
        
        if frame_detections:
            detections.append((frame_idx, frame_detections))
        
        frame_idx += 1
    
    cap.release()
    return detections

def query_frames(detections, query):
    """Find frames matching the query (e.g., 'red car')."""
    query = query.lower().strip()
    try:
        color, vehicle_type = query.split()
    except ValueError:
        return [], "Query must be in format 'color vehicle_type' (e.g., 'red car')"
    
    if color not in color_ranges or vehicle_type not in vehicle_classes:
        return [], f"Invalid color or vehicle type. Supported colors: {list(color_ranges.keys())}, vehicles: {vehicle_classes}"
    
    matching_frames = []
    for frame_idx, frame_detections in detections:
        for detection in frame_detections:
            if detection['vehicle_type'] == vehicle_type and detection['color'] == color:
                matching_frames.append(frame_idx)
                break
    
    return matching_frames, None

def process_video_and_query(video_path, query):
    """Main function to process video and handle query."""
    # Step 1: Detect vehicles directly from video
    detections = detect_vehicles_in_video(video_path, debug=True)
    if not detections:
        return [], "No vehicles detected in video"
    
    # Step 2: Process query
    matching_frames, error = query_frames(detections, query)
    if error:
        return [], error
    
    if not matching_frames:
        return [], f"No frames found with {query}"
    
    return sorted(matching_frames), None

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle detection with color query")
    parser.add_argument("--video", default="D:\\Downloads\\3727445-hd_1920_1080_30fps.mp4", help="Path to video file")
    parser.add_argument("--query", default=None, help="Query in format 'color vehicle_type' (e.g., 'red car')")
    args = parser.parse_args()

    video_path = args.video
    query = args.query

    if not query:
        query = input("Enter query (e.g., 'red car'): ").strip()
    
    if not query:
        print("Error: No query provided")
    else:
        matching_frames, error = process_video_and_query(video_path, query)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Frames containing '{query}': {matching_frames}")