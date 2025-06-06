# Traffic Management System
A Flask-based web application for detecting and analyzing vehicles in traffic videos using YOLOv8 and OpenCV. The system identifies vehicles (cars, trucks, buses, motorcycles) and their colors, allowing users to query specific vehicle types and colors (e.g., "red car") and retrieve matching video frames with timestamps.
Features

## Overview
Vehicle Detection: Uses YOLOv8 (pre-trained on COCO dataset) to detect vehicles in video footage. 🚗🚛🚌🏍️
Color Detection: Identifies vehicle colors (red, blue, green, white, black, yellow, gray, silver) using HSV color space analysis. 🎨
Web Interface: Flask-based UI for uploading videos and querying vehicle types/colors. 🌐
Optimized Processing:
Frame sampling (processes every 3rd frame) for faster analysis. ⚡
Intelligent video resizing for large files. 📹
Enhanced contrast for improved detection accuracy. 🔍
Configurable confidence thresholds for YOLO predictions. 🎯


Output: Saves frames with detected vehicles matching the query, including bounding boxes and timestamps. 🖼️
Debugging: Detailed logging and optional debug frame saving for verification. 🐞

## Requirements

Python 3.8+ 🐍
Dependencies (listed in requirements.txt):
opencv-python
ultralytics
numpy
flask
werkzeug

## Project structure
```
├── app.py # Flask web application (backend)
├── main.py # CLI version for testing detection
├── templates/
│ └── index.html # Web UI
├── static/
│ ├── uploads/ # Uploaded video files
│ └── output_frames/ # Saved result frames with detected matches
```



## Installation

Clone the Repository:
```
git clone https://github.com/your-username/traffic-management-system.git
cd traffic-management-system
```
Set Up a Virtual Environment (optional but recommended):
```
python -m venv venv
```
source venv/bin/activate  # On Windows: venv\Scripts\activate


## Install Dependencies:
```
pip install -r requirements.txt
```

### Download YOLOv8 Model:

The application uses yolov8n.pt (lightweight model). Download it from the Ultralytics YOLOv8 releases or ensure it's available in the project directory. 📥
Optionally, use yolov8m.pt for higher accuracy (slower processing).


Create Static Folders:Ensure the static/uploads and static/output_frames directories exist:
```
mkdir -p static/uploads static/output_frames
```


## Usage
Running the Application

## Start the Flask Server:
```
python app.py
```
The server will run at http://0.0.0.0:5000 by default. 🌍

### Access the Web Interface:

Open a browser and navigate to http://localhost:5000. 🖥️
Upload a traffic video (supported formats: .mp4, .avi, .mov). 📤
Enter a query in the format color vehicle_type (e.g., "red car"). 🔎
Click "Analyze Traffic" to process the video. 🚀


## View Results:

The application displays frames where the specified vehicle and color are detected. 🖼️
Each frame includes a bounding box around the detected vehicle and a timestamp. ⏰



## Running the Standalone Script
For command-line usage without the web interface:

## Run the Script:
```
python main.py --video path/to/video.mp4 --query "red car"
```
Replace path/to/video.mp4 with the path to your video file and "red car" with your desired query. 📜

## Output:

Prints frame numbers where the queried vehicle type and color are detected. 📋
Optionally saves debug images for red vehicles in the current directory (if debug=True). 🖼️



## Project Structure

- main.py: Standalone script for vehicle detection and color analysis from a video file. 🛠️
- app.py: Flask application for the web interface, including video upload and query processing. 🌐
- index.html: HTML template for the web interface, styled with Tailwind CSS. 🎨
- static/uploads/: Directory for uploaded video files. 📤
- static/output_frames/: Directory for saved frames with detected vehicles. 🖼️
- requirements.txt: List of Python dependencies. 📦

## Supported Queries

- **Vehicle Types**: car, truck, bus, motorcycle 🚗🚛🚌🏍️
- **Colors**: red, blue, green, white, black, yellow, gray, silver 🎨
- **Query Format**: color vehicle_type (e.g., red car, blue truck)

## Notes

- **Performance**: The application processes every 3rd frame to optimize speed. Adjust skip_frames in app.py for different sampling rates. ⚡
- **Color Detection**: Uses HSV color ranges for robust detection. The app.py version includes improved ranges for gray and silver. 🎨
- **Debugging**: Set debug=True in main.py or app.py to enable detailed logging and save debug images for red vehicles (in main.py) or matching query frames (in app.py). 🐞
- **Video Formats**: Ensure videos are in supported formats (.mp4, .avi, .mov). Large videos are automatically resized for faster processing. 📹

## Limitations

- Color detection accuracy depends on lighting conditions and video quality. 💡
- YOLOv8's lightweight model (yolov8n.pt) may miss some vehicles; use yolov8m.pt for better accuracy at the cost of speed. ⚖️
- The web interface keeps uploaded videos and output frames until manually deleted (or automatically cleaned on error/no matches). 🗑️

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports or feature requests. 🙌

## Developed by
Mohammad Irfan S
