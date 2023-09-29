import cv2
import numpy as np
import time
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker,STrack
from supervision.tools.detections import Detections, BoxAnnotator
from typing import List
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from onemetric.cv.utils.iou import box_iou_batch
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point


# Load the YOLOv8 model
model = YOLO("yolov8s.pt")
CLASS_NAMES_DICT = model.model.names
SCALE = 0.5


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

tracker = BYTETracker(BYTETrackerArgs())


# Initialize the person counter
person_count = 0

# Open the video file
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("vehicle-counting.mp4")

def resize_frame(frame, scale=SCALE):
  """Resizes a video frame.

  Args:
    frame: A numpy array representing the video frame.
    scale: The scale factor to resize the frame by.

  Returns:
    A numpy array representing the resized video frame.
  """

  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dim = (width, height)
  return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)



box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
line_annotator = LineCounterAnnotator(thickness=2, text_thickness=1, text_scale=1)


# Define the entrance to the room
line_counter = LineCounter(start=Point(0, 600), end=Point(1800, 600))
width  = cap.get(3)   # float `width`
height = cap.get(4)     # float `height`
line_counter = LineCounter(start=Point(0, int(height/2)*SCALE), end=Point(width*SCALE, int(height/2)*SCALE))

# While the video is open
while True:

    # Read the next frame from the video
    ret, frame = cap.read()

    # If the frame is not empty
    if ret:
        # Resize the frame
        frame = resize_frame(frame)

        # Make a copy of the frame
        frame_copy = frame.copy()

     
        # Run YOLOv8 inference on the frame
        results = model(frame_copy)

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

         # Track the detections
        # tracks = tracker.update(detections)
        tracks = tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame_copy.shape,
            img_size=frame_copy.shape
        )

        tracker_ids = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_ids)
        # filtering out detections without trackers
        mask = np.array([tracker_ids is not None for tracker_ids in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_ids} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_ids
            in detections
        ]
        # updating line counter
        line_counter.update(detections=detections)
        print(line_counter.out_count)


        # annotate and display frame
        frame = box_annotator.annotate(frame=frame_copy, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame_copy, line_counter=line_counter)


        # Display the frame
        cv2.imshow("Frame", frame_copy)

        

        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Else, if the frame is empty
    else:

        # Break the loop
        break

# Release the video file
cap.release()

# Close all windows
cv2.destroyAllWindows()
