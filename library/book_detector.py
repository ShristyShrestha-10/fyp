import cv2
import numpy as np
import apriltag
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BookDetector:
    def __init__(self):
        """Initialize the AprilTag detector with the tag family used for books."""
        try:
            # Initialize AprilTag detector with tag41h12 family (good balance of size and reliability)
            self.detector = apriltag.Detector()
            logger.info("AprilTag detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AprilTag detector: {str(e)}")
            self.detector = None

    def detect_books(self, frame):
        """Detect AprilTags in the given frame and return their IDs and locations."""
        try:
            if self.detector is None:
                logger.error("AprilTag detector not initialized")
                return []

            # Handle case where frame is a bytes object (from JPEG compression)
            if isinstance(frame, bytes):
                # Convert bytes to numpy array
                np_arr = np.frombuffer(frame, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.error("Failed to decode frame from bytes")
                    return []

            # Convert frame to grayscale for AprilTag detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Detect AprilTags
            detections = self.detector.detect(gray)
            
            if detections:
                logger.info(f"Detected {len(detections)} AprilTags")
                
                # Process each detection
                results = []
                for detection in detections:
                    # Ensure tag_id is an integer
                    tag_id = int(detection.tag_id)
                    corners = detection.corners
                    center = detection.center
                    
                    # Add detection info to results
                    results.append({
                        'tag_id': tag_id,
                        'corners': corners.tolist(),
                        'center': center.tolist(),
                        'timestamp': datetime.now()
                    })
                    
                    # Draw detection on frame for visualization
                    self._draw_detection(frame, corners, tag_id)
                
                return results
            else:
                logger.debug("No AprilTags detected in frame")
                return []

        except Exception as e:
            logger.error(f"Error detecting AprilTags: {str(e)}")
            return []

    def _draw_detection(self, frame, corners, tag_id):
        """Draw the detected AprilTag and its ID on the frame."""
        try:
            # Convert corners to integer points
            corners = corners.astype(int)
            
            # Draw the tag outline
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            
            # Calculate center point for text
            center_x = int(corners[:, 0].mean())
            center_y = int(corners[:, 1].mean())
            
            # Draw tag ID
            cv2.putText(frame, f"ID: {tag_id}", (center_x - 20, center_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                      
        except Exception as e:
            logger.error(f"Error drawing detection: {str(e)}") 