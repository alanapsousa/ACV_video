"""
Detect a Fall 
- Using 8 points from Mediapipe
- Using 3 criteria
- Strict Confirmation 
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime
import os
import csv
import threading

LOGS_DIR = "./logs"

class FallDetector:
    def __init__(self):
        """Initialize the detector"""
        
        print("⚡ Initializing detector...")
        
        # MediaPipe Pose (same configuration)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Counters
        self.fall_count = 0
        self.current_state = None
        self.state_history = deque(maxlen=5)
        self.alert_active = False
        
        # Minimal history (only height and velocity)
        self.height_history = deque(maxlen=20)  # Reduced: 20 instead of 30
        self.previous_center_y = None
        
        # Strict confirmation
        self.fall_state_counter = 0
        self.fall_confirmed = False
        self.recovery_counter = 0
        self.recovery_delay = 100  # 3 seconds (very long to avoid duplicates)
        
        os.makedirs(LOGS_DIR, exist_ok=True)
        print("✓ Detector initialized (8 points, 3 criteria)")
    
    def calculate_velocity(self, current_y):
        """Calculate vertical velocity"""
        if self.previous_center_y is None:
            self.previous_center_y = current_y
            return 0, 0
        
        velocity = current_y - self.previous_center_y
        self.previous_center_y = current_y
        return abs(velocity), velocity
    
    def analyze_fall(self, landmarks):
        """
        ULTRA-SIMPLIFIED analysis with onl'y 8 points and 3 criteria
        
        Points used (8/33):
        - Shoulders (11, 12): body orientation
        - Hips (23, 24): body center
        - Knees (25, 26): leg flexion
        - Ankles (27, 28): feet position
        
        Criteria (only 3):
        1. Fast downward velocity (PRIORITY)
        2. Very low body height
        3. Horizontal trunk angle
        """
        if not landmarks:
            return 0, "unknown", "", {}
        
        lm = landmarks.landmark
        
        # === EXTRACTION OF 8 ESSENTIAL POINTS ===
        left_shoulder = np.array([lm[11].x, lm[11].y])
        right_shoulder = np.array([lm[12].x, lm[12].y])
        left_hip = np.array([lm[23].x, lm[23].y])
        right_hip = np.array([lm[24].x, lm[24].y])
        left_ankle = np.array([lm[27].x, lm[27].y])
        right_ankle = np.array([lm[28].x, lm[28].y])
        
        # Centers
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        ankle_center = (left_ankle + right_ankle) / 2
        body_center_y = (shoulder_center[1] + hip_center[1]) / 2
        
        # === MINIMAL CALCULATIONS ===
        
        # 1. Velocity
        velocity_mag, velocity_dir = self.calculate_velocity(body_center_y)
        
        # 2. Body height (shoulders → ankles)
        body_height = abs(shoulder_center[1] - ankle_center[1])
        self.height_history.append(body_height)
        
        # 3. Trunk angle (shoulders → hips)
        trunk_vector = hip_center - shoulder_center
        trunk_angle = np.degrees(np.arctan2(abs(trunk_vector[0]), abs(trunk_vector[1])))
        
        # =======================================
        # ULTRA-STRICT DETECTION (3 criteria)
        # =======================================
        fall_score = 0
        debug_info = []
        
        # CRITERION 1: Fast downward velocity (> 0.02 - ULTRA sensitive)
        if velocity_mag > 0.02 and velocity_dir > 0:
            fall_score += 0.40  # 40% of score!
            debug_info.append(f"Fast fall: {velocity_mag:.3f}↓")
        
        # CRITERION 2: Low body (< 0.35 - ULTRA permissive)
        if body_height < 0.35:
            fall_score += 0.30 # 30% score
            debug_info.append(f"Very low: {body_height:.2f}")
        
        # CRITERION 3: Horizontal trunk (> 60° - ULTRA permissive)
        if trunk_angle > 60:
            fall_score += 0.30 # 30% score
            debug_info.append(f"Horizontal: {trunk_angle:.1f}°")
        
        scores = {
            'all': fall_score,
            'velocity': velocity_mag,
            'height': body_height,
            'trunk_angle': trunk_angle
        }
        
        # SENSITIVE decision: Score > 0.50 (ULTRA permissive)
        if fall_score > 0.65:
            state = "fall"
            description = " + ".join(debug_info)
        else:
            state = "safe"
            description = ""
        
        return fall_score, state, description, scores
    
    def detect_landmarks(self, frame):
        """Detect landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks if results and results.pose_landmarks else None
    
    def play_alert(self):
        """Trigger audio alert"""
        def beep():
            try:
                result = os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga 2>/dev/null')
                if result != 0:
                    print('\a' * 4)
            except:
                print('\a' * 4)
        thread = threading.Thread(target=beep, daemon=True)
        thread.start()
    
    def log_fall(self, frame, analysis):
        """Log the fall"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        csv_path = os.path.join(LOGS_DIR, "fall_log_ultra_simple.csv")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if os.path.getsize(csv_path) == 0:
                writer.writerow(["Timestamp", "Fall_Number", "Fall_Score", "Details"])
            writer.writerow([timestamp, self.fall_count, f"{analysis['all']:.2f}", analysis.get('description', '')])
        
        img_path = os.path.join(LOGS_DIR, f"fall_ultra_{self.fall_count}_{timestamp.replace(':', '-')}.jpg")
        cv2.imwrite(img_path, frame)
    
    def update(self, frame):
        """Update with strict logic"""
        landmarks = self.detect_landmarks(frame)
        
        if landmarks:
            fall_score, state, description, scores = self.analyze_fall(landmarks)
        else:
            fall_score, state, description, scores = 0, "unknown", "", {"all": 0}
        
        self.state_history.append(state)
        
        # Decrement recovery
        if self.recovery_counter > 0:
            self.recovery_counter -= 1
        
        # Majority vote (at least 3/5)
        if len(self.state_history) >= 3:
            fall_votes = sum(1 for s in self.state_history if s == "fall")
            safe_votes = sum(1 for s in self.state_history if s == "safe")
            
            if fall_votes >= 2:  # Softened: 2/5 votes are sufficient
                self.current_state = "fall"
                self.fall_state_counter += 1
            elif safe_votes >= 2:
                self.current_state = "safe"
                self.fall_state_counter = 0
                if self.recovery_counter == 0:
                    self.fall_confirmed = False
        else:
            self.current_state = state
            if state == "fall":
                self.fall_state_counter += 1
            else:
                self.fall_state_counter = 0
        
        # FAST CONFIRMATION: 3 consecutive frames (0.10 sec - ultra fast)
        if (self.current_state == "fall" and 
            self.fall_state_counter >= 3 and 
            not self.fall_confirmed and 
            self.recovery_counter == 0):
            
            self.fall_confirmed = True
            self.alert_active = True
            self.fall_count += 1
            self.recovery_counter = self.recovery_delay
            
            self.play_alert()
            self.log_fall(frame, {"description": description, "all": fall_score})
            print(f"🚨 FALL DETECTED! #{self.fall_count}")
            print(f"   Score: {fall_score:.2f} | {description}")
        
        if self.current_state == "safe" and self.recovery_counter == 0:
            self.alert_active = False
            self.fall_confirmed = False
        
        return {
            "fall_count": self.fall_count,
            "state": self.current_state,
            "fall_score": fall_score,
            "description": description,
            "scores": scores,
            "landmarks": landmarks
        }
    
    def draw_landmarks(self, frame, landmarks):
        """Draw skeleton"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
            )
        return frame
    
    def draw_info(self, frame, info):
        """Display information"""
        frame = self.draw_landmarks(frame, info["landmarks"])
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Display Text
        y = 25
        #cv2.putText(frame, f"FALLS: {info['fall_count']}", (15, y), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        color = (0, 0, 255) if info['state'] == 'fall' else (0, 255, 0)
        cv2.putText(frame, f"STATE: {info['state'].upper()}", (15, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.putText(frame, f"SCORE: {info['fall_score']:.2f}", (15, y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # Badge "ULTRA-SIMPLE"
        #cv2.putText(frame, "(8 pts, 3 criteria)", (450, 40), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        if info['description']:
            cv2.putText(frame, info['description'][:60], (15, y + 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Alert
        if info['state'] == 'fall':
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
            cv2.putText(frame, "!!! FALL DETECTED !!!", (200, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()

def run_detector():
    """Launch the detector"""
    print("=" * 70)
    print("🚨 GEOMETRY FALL DETECTOR")
    #print("   - 8 detection points (instead of 13)")
    #print("   - 3 essential criteria (instead of 5)")
    #print("   - Ultra-strict threshold: 0.75")
    #print("   - Recovery delay: 8 seconds")
    print("=" * 70)
    
    detector = FallDetector()
    camera_index = 2
    cap = cv2.VideoCapture(camera_index)  # External camera
    
    if not cap.isOpened():
        print(f"❌ Camera {camera_index} is not available!")
        return
    
    print("\n✓ Camera Opened")
    print("- Press 'q' to quit")
    print("-" * 70)
    
    # Create a named window 
    window_name = "Fall Detector - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
        
            frame = cv2.resize(frame, (1280, 720))
            info = detector.update(frame)
            frame_count += 1
        
            #if frame_count % 10 == 0:
            #    print(f"Frame {frame_count} | Falls: {info['fall_count']} | "
            #          f"State: {info['state']} | Score: {info['fall_score']:.2f}")
        
            frame = detector.draw_info(frame, info)
            cv2.imshow(window_name, frame)
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.fall_count = 0
                print("✓ Fall count reset to 0")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()

    
    print("\n" + "=" * 70)
    print(f"✓ Detection completed")
    print(f"Total falls: {detector.fall_count}")
    print(f"Logs: {LOGS_DIR}/fall_log_ultra_simple.csv")
    print("=" * 70)


if __name__ == "__main__":
    run_detector()
