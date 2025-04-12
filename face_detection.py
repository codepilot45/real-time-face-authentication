import cv2
import os

print("📁 Running from:", os.getcwd())

# ✅ Load trained model
if not os.path.exists("classifier.xml"):
    raise FileNotFoundError("❌ classifier.xml not found. Run classifier.py first.")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# ✅ Only recognize Siva (ID 1)
id_to_name = {
    1: "Siva",
    2: "Harshit"  # still here for completeness, but not used
}
allowed_ids = {1}  # ✅ Only allow Siva's face to be recognized

# ✅ Load Haar cascade
def load_cascade(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Cascade file not found: {path}")
    return cv2.CascadeClassifier(path)

faceCascade = load_cascade('haarcascade_frontalface_default.xml')

# ✅ Recognition & display
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (200, 200))
        roi_normalized = cv2.equalizeHist(roi_resized)

        id, confidence = clf.predict(roi_normalized)
        match_percentage = 100 - confidence

        if id in allowed_ids and match_percentage > 40:
            label = id_to_name[id]
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        confidence_text = f"{match_percentage:.2f}% Match"

        # Draw UI
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, confidence_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return img

# ✅ Webcam loop
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise IOError("❌ Cannot open webcam")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to capture image")
        break

    frame = draw_boundary(frame, faceCascade, 1.1, 10, (255, 255, 255), clf)
    cv2.imshow("🎥 Face Detection & Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
