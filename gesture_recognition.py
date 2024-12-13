import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from datetime import datetime, timedelta
import json
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe objects correctly
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class GestureRecognizerApp:
    def __init__(self):
        self.closed_fist_count = 0
        self.palm_count = 0
        self.thumbup_count = 0
        self.pointup_count = 0
        self.previous_gesture = "No gesture"
        
        # Initialize MediaPipe drawing utilities
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize hand landmarks
        self.current_hand_landmarks = None
        
        # Add English alphabet and letter tracking
        self.polish_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.current_letter_index = 0
        self.last_letter_change = None
        self.selected_letters = []
        self.letter_display_interval = 1.0
        
        # Add timing variables
        self.gesture_start_time = None
        self.current_hold_duration = timedelta()
        
        # Add cleanup handler
        self._recognizer = None
        
        # Initialize gesture recognizer following documentation pattern
        def result_callback(result: GestureRecognizerResult, 
                          output_image: mp.Image, 
                          timestamp_ms: int) -> None:
            if result.gestures:
                gesture_name = result.gestures[0][0].category_name
                if gesture_name != self.previous_gesture:
                    self.handle_gesture_change(gesture_name)
                self.previous_gesture = gesture_name
            
            # Store the latest hand landmarks
            self.current_hand_landmarks = result.hand_landmarks if result.hand_landmarks else None

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_hands=1)
        
        self._recognizer = GestureRecognizer.create_from_options(options)

    def __del__(self):
        if self._recognizer:
            self._recognizer.close()

    def handle_gesture_change(self, gesture_name):
        current_time = datetime.now()
        
        if gesture_name == "Open_Palm":
            self.palm_count += 1
            if self.current_letter_index < len(self.polish_alphabet):
                self.selected_letters.append(self.polish_alphabet[self.current_letter_index])
        elif gesture_name == "Closed_Fist":
            self.closed_fist_count += 1
            self.selected_letters.append(" ")
        elif gesture_name == "Thumb_Up":
            self.thumbup_count += 1
            if self.selected_letters:
                word = ''.join(self.selected_letters)
                log_data = {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "sent": f"SENT: {word}"
                }
                print(json.dumps(log_data))
                self.selected_letters = []
        elif gesture_name == "Pointing_Up":
            self.pointup_count += 1
            if self.selected_letters:
                log_data = {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "reset": "WORD RESET"
                }
                print(json.dumps(log_data))
            self.selected_letters = []
            self.current_letter_index = 0
            self.last_letter_change = current_time

        # Update JSON logging
        log_data = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "gesture": gesture_name,
            "closed_fist_counter": self.closed_fist_count,
            "openpalm_counter": self.palm_count,
            "thumbup_counter": self.thumbup_count,
            "pointup_counter": self.pointup_count
        }
        print(json.dumps(log_data))

    def process_frame(self, frame):
        current_time = datetime.now()
        
        # Handle letter iteration
        if self.last_letter_change is None:
            self.last_letter_change = current_time
        else:
            time_since_last_change = (current_time - self.last_letter_change).total_seconds()
            if time_since_last_change >= self.letter_display_interval:
                self.current_letter_index = (self.current_letter_index + 1) % len(self.polish_alphabet)
                self.last_letter_change = current_time

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        
        # Process the frame
        self._recognizer.recognize_async(mp_image, int(current_time.timestamp() * 1000))

        # Draw hand landmarks if available
        if self.current_hand_landmarks and len(self.current_hand_landmarks) > 0:
            for hand_landmarks in self.current_hand_landmarks:

                # Draw connections between landmarks
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_point = (int(hand_landmarks[start_idx].x * frame.shape[1]),
                                 int(hand_landmarks[start_idx].y * frame.shape[0]))
                    end_point = (int(hand_landmarks[end_idx].x * frame.shape[1]),
                               int(hand_landmarks[end_idx].y * frame.shape[0]))
                    
                    # Increased line thickness to 3 and using pure green
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 3)
                
                # Draw each landmark point
                for landmark in hand_landmarks:
                    # Get coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    # Draw circle at landmark position - outer green circle and inner red circle
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)  # Pure green (0,255,0), radius 15
                    cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)  # Pure red (0,0,255), radius 12
                

        # Display current gesture (top left)
        cv2.putText(
            frame,
            f"Current Gesture: {self.previous_gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Display counters (bottom right)
        counter_text = f"CLOSED_FIST: {self.closed_fist_count} | OPEN_PALM: {self.palm_count} | THUMB_UP: {self.thumbup_count} | POINT_UP: {self.pointup_count}"
        height, width = frame.shape[:2]
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = width - text_size[0] - 10
        text_y = height - 20
        cv2.putText(
            frame,
            counter_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Display current letter in center
        current_letter = self.polish_alphabet[self.current_letter_index]
        letter_size = cv2.getTextSize(current_letter, cv2.FONT_HERSHEY_SIMPLEX, 12, 6)[0]
        letter_x = (width - letter_size[0]) // 2
        letter_y = (height + letter_size[1]) // 2
        cv2.putText(
            frame,
            current_letter,
            (letter_x, letter_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            12,
            (255, 255, 255),
            12
        )

        # Display selected letters at bottom center
        if self.selected_letters:
            word = ''.join(self.selected_letters)
            word_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
            word_x = (width - word_size[0]) // 2
            word_y = height - 50
            cv2.putText(
                frame,
                word,
                (word_x, word_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (255, 255, 255),
                4
            )

        return frame

def main():
    cap = cv2.VideoCapture(0)
    recognizer = GestureRecognizerApp()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = recognizer.process_frame(frame)
            cv2.imshow('Gesture Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 