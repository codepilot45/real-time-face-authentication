import cv2
import os

# ✅ Fixed user details
user_id = 1
user_name = "siva"

# ✅ Create directory for Siva's images
user_dir = f"dataset/{user_id}_{user_name}"
os.makedirs(user_dir, exist_ok=True)

def generate_dataset(img, id, img_id):
    filename = os.path.join(user_dir, f"user.{id}.{img_id}.jpg")
    success = cv2.imwrite(filename, img)
    print(f"[INFO] Saved {filename}: {success}")

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, img_id):
    color = {"blue": (255, 0, 0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        generate_dataset(roi_img, user_id, img_id)
    return img

# ✅ Load Haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceCascade.empty():
    print("[ERROR] Could not load face cascade!")
    exit()

# ✅ Open camera
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

# ✅ Image capture loop
img_id = 0
max_images = 100

while True:
    ret, img = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
    img = detect(img, faceCascade, img_id)
    cv2.imshow(f"Capturing for {user_name} - Press Q to Quit", img)
    img_id += 1
    if img_id % 10 == 0:
        print(f"[INFO] Collected {img_id} images for {user_name}")
    if cv2.waitKey(1) & 0xFF == ord('q') or img_id >= max_images:
        break

video_capture.release()
cv2.destroyAllWindows()
