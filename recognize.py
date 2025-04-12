import cv2
import os

# Print current working directory for debug
print("📁 Running from:", os.getcwd())

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords.append([x, y, w, h])
    return coords, img

def detect(img, faceCascade, eyeCascade, leftEyeCascade, noseCascade):
    colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "yellow": (0, 255, 255)}

    coords, img = draw_boundary(img, faceCascade, 1.1, 10, colors["blue"], "Face")

    for (x, y, w, h) in coords:
        roi_color = img[y:y + h, x:x + w]
        draw_boundary(roi_color, eyeCascade, 1.1, 10, colors["green"], "Right Eye")
        draw_boundary(roi_color, leftEyeCascade, 1.1, 10, colors["green"], "Left Eye")
        draw_boundary(roi_color, noseCascade, 1.1, 10, colors["yellow"], "Nose")

    return img

# Load cascades using absolute paths
base_path = r"C:\Users\2349s\PycharmProjects\PythonProject\Face-Detection-Recognition-Using-OpenCV-in-Python"
faceCascade = cv2.CascadeClassifier(os.path.join(base_path, "haarcascade_frontalface_default.xml"))
eyeCascade = cv2.CascadeClassifier(os.path.join(base_path, "haarcascade_eye.xml"))
leftEyeCascade = cv2.CascadeClassifier(os.path.join(base_path, "haarcascade_lefteye_2splits.xml"))
noseCascade = cv2.CascadeClassifier(os.path.join(base_path, "haarcascade_mcs_nose.xml"))

# ✅ Debug info to find what's missing
if faceCascade.empty(): print("❌ Face cascade NOT loaded")
if eyeCascade.empty(): print("❌ Eye cascade NOT loaded")
if leftEyeCascade.empty(): print("❌ Left eye cascade NOT loaded")
if noseCascade.empty(): print("❌ Nose cascade NOT loaded")

if any(cascade.empty() for cascade in [faceCascade, eyeCascade, leftEyeCascade, noseCascade]):
    raise IOError("One or more Haar Cascade XML files not found")

# Open webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, img = video_capture.read()
    if not ret:
        break

    img = detect(img, faceCascade, eyeCascade, leftEyeCascade, noseCascade)
    cv2.imshow("Face Feature Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
