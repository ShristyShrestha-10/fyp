import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import time
import face_recognition

class IDCardScanner:
    def __init__(self):
        # Initialize YOLO model
        try:
            self.yolo_model = YOLO("yolo11n.pt")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")

        # Get camera dimensions
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read from camera")
        self.height, self.width = frame.shape[:2]

        # Calculate ID card rectangle dimensions for left side placement
        self.card_ratio = 85.6/54.0
        self.rect_height = int(self.height * 0.6)
        self.rect_width = int(self.rect_height / self.card_ratio)
        # Change x position to left side (with small margin)
        self.x = 50  # 50 pixels from left edge
        self.y = (self.height - self.rect_height) // 2

        # Create media directory if it doesn't exist
        self.media_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')
        self.faces_dir = os.path.join(self.media_dir, 'faces')
        self.cards_dir = os.path.join(self.media_dir, 'cards')
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.cards_dir, exist_ok=True)

        # Detection tracking variables
        self.face_detected_time = None
        self.card_detected_time = None
        self.images_saved = False
        self.detection_countdown = 3  # 3 seconds countdown

        # Add similarity threshold
        self.similarity_threshold = 0.6
        self.face_encoding = None
        self.id_card_face_encoding = None

    def extract_face_from_id_card(self, frame):
        # Extract face region from ID card area
        card_area = frame[self.y:self.y+self.rect_height, self.x:self.x+self.rect_width]
        # Convert to RGB for face_recognition library
        rgb_card = cv2.cvtColor(card_area, cv2.COLOR_BGR2RGB)
        # Detect faces in ID card
        face_locations = face_recognition.face_locations(rgb_card)
        
        if face_locations:
            # Get face encoding from ID card
            self.id_card_face_encoding = face_recognition.face_encodings(rgb_card, face_locations)[0]
            return True
        return False

    def check_face_similarity(self, frame, face_bbox):
        if face_bbox is None or self.id_card_face_encoding is None:
            return False
            
        x, y, w, h = face_bbox
        face_image = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Get face encoding for the detected face
        face_locations = face_recognition.face_locations(rgb_face)
        if not face_locations:
            return False
            
        live_face_encoding = face_recognition.face_encodings(rgb_face, face_locations)[0]
        
        # Compare faces
        distance = face_recognition.face_distance([self.id_card_face_encoding], live_face_encoding)[0]
        similarity = 1 - distance
        
        return similarity >= self.similarity_threshold

    def save_images(self, frame, face_bbox, card_bbox):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate a more meaningful filename
        filename_base = f"verification_{timestamp}"
        
        # Save face image
        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_img = frame[y:y+h, x:x+w]
            face_path = os.path.join(self.faces_dir, f'{filename_base}_live.jpg')
            cv2.imwrite(face_path, face_img)

        # Save ID card image
        if card_bbox is not None:
            x, y, w, h = card_bbox
            card_img = frame[y:y+h, x:x+w]
            card_path = os.path.join(self.cards_dir, f'{filename_base}_card.jpg')
            cv2.imwrite(card_path, card_img)

        return True

    def process_frame(self, frame):
        current_time = time.time()
        face_bbox = None
        card_bbox = None
        face_detected = False
        card_detected = False

        # YOLO person detection
        if self.yolo_model:
            results = self.yolo_model(frame)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face_bbox = (x, y, w, h)
                face_detected = True
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Face: {detection.score[0]:.2f}',
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract face from ID card when detected
        if face_detected and self.id_card_face_encoding is None:
            self.extract_face_from_id_card(frame)

        # Check face similarity and handle capture
        if face_detected and card_detected and not self.images_saved:
            if face_bbox is not None:
                is_same_person = self.check_face_similarity(frame, face_bbox)
                
                if is_same_person:
                    if self.face_detected_time is None:
                        self.face_detected_time = current_time
                    
                    time_detected = current_time - self.face_detected_time
                    
                    if time_detected >= self.detection_countdown:
                        if not self.images_saved:
                            self.images_saved = self.save_images(frame, face_bbox, card_bbox)
                            cv2.putText(frame, 'Face Verified & Images Captured!', 
                                      (int(self.width/2) - 150, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    # Reset timer if faces don't match
                    self.face_detected_time = None
                    if not self.images_saved:
                        cv2.putText(frame, 'Face Does Not Match ID Card', 
                                  (int(self.width/2) - 150, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.face_detected_time = None

        # Draw ID card rectangle guide in blue
        cv2.rectangle(frame, (self.x, self.y),
                     (self.x + self.rect_width, self.y + self.rect_height),
                     (255, 0, 0), 2)  # Changed to blue (BGR format)

        # Add guiding text only if images haven't been captured
        if not self.images_saved:
            text = "Place ID Card Here"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = self.x + (self.rect_width - text_size[0]) // 2
            text_y = self.y - 10
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                       (255, 0, 0), font_thickness)  # Changed to blue
        
        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow('ID Card Scanner', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or self.images_saved:
                    # Wait for 2 seconds after saving images before closing
                    if self.images_saved:
                        time.sleep(2)
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        scanner = IDCardScanner()
        scanner.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()