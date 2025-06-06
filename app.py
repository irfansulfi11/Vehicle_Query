import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import glob
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output_frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
    logging.info("YOLOv8 model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load YOLOv8 model: {e}")
    raise e

# Define vehicle classes and IMPROVED color ranges for better accuracy
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
class_ids = [2, 5, 7, 3]  # COCO class IDs

# More precise color ranges for better accuracy
color_ranges = {
    'red': [((0, 50, 50), (10, 255, 255)), ((160, 50, 50), (180, 255, 255))],  # True reds
    'blue': [((90, 50, 50), (130, 255, 255))],  # True blues
    'green': [((40, 50, 50), (80, 255, 255))],  # True greens
    'white': [((0, 0, 180), (180, 30, 255))],  # True whites
    'black': [((0, 0, 0), (180, 255, 50))],  # True blacks
    'yellow': [((15, 50, 50), (35, 255, 255))],  # True yellows
    'gray': [((0, 0, 50), (180, 30, 180)),  # Gray range
             ((0, 0, 100), (180, 30, 200))],  # Light gray
    'silver': [((0, 0, 120), (180, 25, 220))]  # Silver cars
}

# Color-specific bounding box colors for debug images (BGR)
debug_box_colors = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'white': (255, 255, 255),
    'black': (100, 100, 100),
    'yellow': (0, 255, 255)
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dominant_color(image, box, debug=False):
    x1, y1, x2, y2 = map(int, box)
    # Crop to central 80% of the bounding box
    margin = 0.1
    width, height = x2 - x1, y2 - y1
    x1_new = x1 + int(width * margin)
    y1_new = y1 + int(height * margin)
    x2_new = x2 - int(width * margin)
    y2_new = y2 - int(height * margin)
    roi = image[y1_new:y2_new+1, x1_new:x2_new+1]
    if roi.size == 0:
        if debug:
            logging.warning(f"Empty ROI for box {box}")
        return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_roi, axis=(0, 1))
    if debug:
        logging.debug(f"Box {box}, Cropped ROI Mean HSV: {mean_hsv}")
    for color_name, ranges in color_ranges.items():
        for lower, upper in ranges:
            if (lower[0] <= mean_hsv[0] <= upper[0] and
                lower[1] <= mean_hsv[1] <= upper[1] and
                lower[2] <= mean_hsv[2] <= upper[2]):
                if debug:
                    logging.debug(f"Detected color: {color_name} for box {box}")
                return color_name
    if debug:
        logging.debug(f"No color match for box {box}, HSV: {mean_hsv}")
    return None

def detect_vehicles_in_video(video_path, query, confidence_threshold=0.25, debug=False):
    logging.info(f"Processing video: {video_path}, Query: {query}")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Failed to open video file")
            raise ValueError("Error opening video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video: {fps} FPS, {total_frames} total frames")
        
        try:
            color, vehicle_type = query.lower().split()
        except ValueError:
            cap.release()
            logging.error("Invalid query format")
            return [], "Query must be in format 'color vehicle_type' (e.g., 'red car')", None

        if color not in color_ranges or vehicle_type not in vehicle_classes:
            cap.release()
            logging.error(f"Invalid color or vehicle type: {color}, {vehicle_type}")
            return [], f"Invalid color or vehicle type. Supported colors: {list(color_ranges.keys())}, vehicles: {vehicle_classes}", None

        # Clean up old images in output folder
        try:
            for old_file in glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], '*.jpg')):
                os.remove(old_file)
                logging.debug(f"Removed old image: {old_file}")
        except Exception as e:
            logging.error(f"Failed to clean output folder: {e}")

        image_data = []
        frame_idx = 0
        matches_found = 0
        
        # SPEED OPTIMIZATION: Skip frames (process every 3rd frame for faster processing)
        skip_frames = 3
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for speed (only process every 3rd frame)
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
                
            # Resize frame for faster processing (optional - comment out if quality is important)
            height, width = frame.shape[:2]
            if width > 1280:  # Only resize if video is very large
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Light contrast enhancement (reduced processing)
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            
            # Run YOLO with higher confidence threshold for speed
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            frame_has_match = False
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids_detected = result.boxes.cls.cpu().numpy()
                
                for box, score, cls_id in zip(boxes, scores, class_ids_detected):
                    if int(cls_id) in class_ids:
                        detected_vehicle = vehicle_classes[class_ids.index(int(cls_id))]
                        color_detected = get_dominant_color(frame, box, debug=debug)
                        
                        # ONLY save frames that match the query exactly
                        if color_detected == color and detected_vehicle == vehicle_type:
                            if not frame_has_match:  # Only save the frame once even if multiple matches
                                # Calculate actual timestamp (accounting for skipped frames)
                                actual_frame = frame_idx
                                timestamp = actual_frame / fps
                                minutes, seconds = divmod(int(timestamp), 60)
                                milliseconds = int((timestamp % 1) * 1000)
                                timestamp_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                                
                                # Create a copy of the frame for saving
                                output_frame = frame.copy()
                                
                                # Draw bounding box for the matching vehicle
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                
                                # Add label showing the matched query
                                label = f"MATCH: {color} {detected_vehicle} - {timestamp_str}"
                                cv2.putText(output_frame, label, (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                # Generate filename indicating this is a query match
                                image_filename = f"match_{color}_{detected_vehicle}_{frame_idx}_{matches_found}.jpg"
                                image_path = os.path.join(app.config['OUTPUT_FOLDER'], image_filename)
                                
                                # Ensure the directory exists
                                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                                
                                # Save ONLY the matching frame
                                success = cv2.imwrite(image_path, output_frame)
                                if success:
                                    logging.info(f"✓ QUERY MATCH SAVED: {image_path}")
                                    image_data.append((image_filename, timestamp_str))
                                    matches_found += 1
                                    frame_has_match = True
                                else:
                                    logging.error(f"Failed to save query match: {image_path}")
            
            frame_idx += 1
            
            # Progress logging every 100 frames
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                logging.info(f"Processing progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
            
        cap.release()
        
        if not image_data:
            logging.info(f"No frames matched query: {query}")
        else:
            logging.info(f"Found {len(image_data)} matching frames for query: {query}")
            
        return image_data, None, os.path.basename(video_path)
        
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        if 'cap' in locals():
            cap.release()
        return [], str(e), None

@app.route('/')
def index():
    return render_template('index.html')

# Add route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/process', methods=['POST'])
def process_video():
    logging.info("Received /process request")
    if 'video' not in request.files or 'query' not in request.form:
        logging.error("Missing video file or query")
        return jsonify({'error': 'Missing video file or query'}), 400
    
    file = request.files['video']
    query = request.form['query'].strip()
    
    if file.filename == '' or not query:
        logging.error("No video file or query provided")
        return jsonify({'error': 'No video file or query provided'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(video_path)
            logging.info(f"Saved uploaded video: {video_path}")
        except Exception as e:
            logging.error(f"Failed to save video {video_path}: {e}")
            return jsonify({'error': f"Failed to save video: {e}"}), 500
        
        try:
            image_data, error, video_filename = detect_vehicles_in_video(video_path, query, debug=False)
            
            if error:
                # Clean up uploaded video on error
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logging.debug(f"Removed video on error: {video_path}")
                logging.error(f"Processing error: {error}")
                return jsonify({'error': error}), 400
            
            if not image_data:
                # Clean up uploaded video when no matches found
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logging.debug(f"Removed video on no detections: {video_path}")
                logging.info(f"No frames found for query: {query}")
                return jsonify({'error': f"No frames found with {query}"}), 400
            
            # Keep the video file for user reference, clean up later if needed
            # os.remove(video_path)  # Uncomment if you want to delete after processing
            
            return jsonify({
                'images': [
                    {'url': f"/static/output_frames/{img}", 'timestamp': ts}
                    for img, ts in image_data
                ],
                'video': f"/static/uploads/{filename}",
                'message': f"Found {len(image_data)} matching frames"
            })
            
        except Exception as e:
            # Clean up uploaded video on exception
            if os.path.exists(video_path):
                os.remove(video_path)
                logging.debug(f"Removed video on exception: {video_path}")
            logging.error(f"Unexpected error: {e}")
            return jsonify({'error': str(e)}), 500
    
    logging.error("Invalid file format")
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)