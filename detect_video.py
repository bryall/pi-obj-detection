# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
from flask import Response
from flask import Flask
from flask import render_template
import threading
import datetime

# initialize the output frame and a lock used to ensure thread-safe exchanges of the output frames (useful when multiple browsers/tabs are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src="http://pumpkinpi.local:8000/video_feed").start()

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_objects(model, labels):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it to have a maximum width of 500 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()

        # prepare the frame for object detection by converting (1) it from BGR to RGB channel ordering and then (2) from a NumPy array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # make predictions on the input frame
        start = time.time()
        results = model.DetectWithImage(frame, threshold=args["confidence"], keep_aspect_ratio=True, relative_coord=False)
        end = time.time()

        # loop over the results
        for r in results:
            # extract the bounding box and box and predicted class label
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            label = labels[r.label_id]

            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv2.putText(orig, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(orig, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # acquire the lock, set the output frame, and release the lock
        with lock:
             outputFrame = orig.copy()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media type (mime type)
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", type=int, required=True, help="port the flask app should run om")
    ap.add_argument("-m", "--model", required=True, help="path to TensorFlow Lite object detection model")
    ap.add_argument("-l", "--labels", required=True, help="path to labels file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize the labels dictionary
    print("[INFO] parsing class labels...")
    labels = {}
    # loop over the class labels file
    for row in open(args["labels"]):
        # unpack the row and update the labels dictionary
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()

    # load the Google Coral object detection model
    print("[INFO] loading Coral model...")
    model = DetectionEngine(args["model"])

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_objects, args=(model, labels,))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host="0.0.0.0", port=args["port"], debug=True, threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()

