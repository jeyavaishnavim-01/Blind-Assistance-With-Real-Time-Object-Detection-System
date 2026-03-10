# STEP 1: IMPORTING LIBRARIES
import cv2                        # OpenCV: Used for image processing and camera handling
import time                       # Time: Used to track performance speed (ms)
import threading                  # Threading: Allows the voice to speak without pausing the video
import queue                      # Queue: Manages the order of audio messages
from ultralytics import YOLO      # YOLO: The Artificial Intelligence model that "sees" objects
import pyttsx3                    # pyttsx3: Offline text-to-speech converter
from collections import Counter   # Counter: Quickly counts duplicates (e.g., "2 persons")

# -------------------------- AUDIO ENGINE CLASS --------------------------
# This class handles the "Voice" of the system in a separate background thread.
class VoiceAssistant:
    def __init__(self):
        self.q = queue.Queue()    # Create a waiting list for messages to be spoken
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 175) # Speed of speech
            # DAEMON THREAD: Runs in the background and closes when the main program stops
            threading.Thread(target=self._run_voice, daemon=True).start()
        except Exception as e:
            print(f"[ERROR] Audio Engine failed to start: {e}")

    def _run_voice(self):
        """ The worker function that constantly checks the queue for text to speak """
        while True:
            text = self.q.get()      # Get the next text from the queue
            self.engine.say(text)    # Load the text into the speech buffer
            self.engine.runAndWait() # Speak the text aloud
            self.q.task_done()       # Mark the task as finished

    def speak(self, text):
        """ Adds text to the queue. If messages are piling up, it clears old ones. """
        if self.q.qsize() > 1:
            with self.q.mutex:
                self.q.queue.clear() # Prevent "Audio Backlog" (hearing things too late)
        self.q.put(text)

# -------------------------- MAIN DETECTION SYSTEM --------------------------
def main():
    print("[INFO] Initializing AI... please wait.")
    
    # 1. LOAD MODEL: 'm' stands for Medium. Balance between accuracy and speed.
    try:
        model = YOLO("yolov8m.pt") 
    except Exception as e:
        print(f"[ERROR] Could not load YOLO model: {e}")
        return

    voice = VoiceAssistant()
    
    # 2. CAMERA SETUP: CAP_DSHOW is for faster camera initialization on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Set Resolution: 1280x720 (HD) for better detection of small objects
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_spoken = {}       # Memory: Stores { 'object_name': last_time_announced }
    remind_interval = 3    # Cooldown: Don't repeat the same object for 3 seconds

    print("\n===== Blind Assistance System With Real Time Object Detection Started =====")
    print("Press 'ESC' on your keyboard to stop.\n")

    while True:
        start_time = time.time()  # Start the clock for this specific frame
        success, frame = cap.read()
        
        if not success:
            print("[WARNING] Failed to grab frame from camera.")
            break

        # 3. DETECTION: conf=0.45 means ignore anything the AI is less than 45% sure about
        results = model(frame, conf=0.45, verbose=False)
        
        current_frame_labels = []   # List for log counts (e.g., "person")
        detailed_labels = []        # List for audio guidance (e.g., "person front")
        h, w, _ = frame.shape       # Get frame dimensions for location logic

        for r in results:
            for box in r.boxes:
                # Get the class ID and look up its name (e.g., 0 -> 'person')
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                
                # 4. LOCATION LOGIC: Divide the frame into 3 vertical zones
                cx = (x1 + x2) // 2       # Find the horizontal center of the object
                if cx < w/3:
                    pos = "left"
                elif cx < 2*w/3:
                    pos = "front"
                else:
                    pos = "right"
                
                current_frame_labels.append(label)
                detailed_labels.append(f"{label} {pos}")

                # 5. VISUALS: Draw the thin green box and label on the screen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 6. LOGGING & AUDIO GUIDANCE
        inference_ms = (time.time() - start_time) * 1000 # Calculate total processing time
        if current_frame_labels:
            # Generate the requested log format
            counts = Counter(current_frame_labels)
            count_str = ", ".join([f"{v} {k}" for k, v in counts.items()])
            print(f"0: {h}x{w} {count_str}, {inference_ms:.1f}ms")
            
            # Smart Audio: Only speak if the object is NEW or has stayed for 3+ seconds
            now = time.time()
            to_say = []
            for item in detailed_labels:
                if item not in last_spoken or (now - last_spoken[item] > remind_interval):
                    to_say.append(item)
                    last_spoken[item] = now
            
            if to_say:
                voice.speak(", ".join(to_say))

        # 7. MEMORY CLEANUP: Forget objects that left the frame so they can be re-announced
        last_spoken = {k: v for k, v in last_spoken.items() if k in detailed_labels}

        # 8. DISPLAY: Show the result in a window
        cv2.imshow("Blind Assistance", frame)
        if cv2.waitKey(1) == 27: # 27 is the ASCII code for the 'ESC' key
            break

    # Clean up resources before closing
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()