import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knn_model.npz")

# กรอบที่ให้เอาหน้าไปวาง (สัดส่วนประมาณ 3:4)
X1, Y1, X2, Y2 = 220, 100, 420, 360

K = 3
THRESHOLD = 400   # ค่าความต่างเฉลี่ยต่อพิกเซล (0-255) ถ้ามากกว่านี้ = Unknown

model = np.load(MODEL_PATH)
X_train = model["X"].astype(np.float32)
y_train = model["y"]
IMG_SIZE = tuple(int(v) for v in model["img_size"])

# ถ้าบางคนมีรูปน้อยกว่า K ให้ลด K ลง (เช่น คนละ 1 รูป -> K = 1)
_, per_person = np.unique(y_train, return_counts=True)
K = max(1, min(K, int(per_person.min())))
print(f"ใช้ K = {K}")


def knn_predict(sample, k=K):
    # ระยะทางแบบเฉลี่ยต่อพิกเซล ตัวเลขจะอยู่ราว 0-255 ไม่ขึ้นกับขนาดรูป
    dists = np.linalg.norm(X_train - sample, axis=1) / np.sqrt(sample.size)
    idx = np.argsort(dists)[:k]
    labels, counts = np.unique(y_train[idx], return_counts=True)
    return labels[np.argmax(counts)], dists[idx].mean()


cap = cv2.VideoCapture(0)
print("วางหน้าในกรอบสีแดง กด q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        print('เปิดกล้องไม่ได้')
        break

    # แปลงทั้งภาพเป็นขาวดำ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตัดภาพเฉพาะในกรอบ แล้วเตรียมข้อมูลแบบเดียวกับตอนเทรน
    face = gray[Y1:Y2, X1:X2]
    face = cv2.equalizeHist(cv2.resize(face, IMG_SIZE))
    sample = face.flatten().astype(np.float32)

    label, dist = knn_predict(sample)
    if dist > THRESHOLD:
        label = "Unknown"

    # แปลงกลับเป็น 3 channel เพื่อให้กรอบและตัวหนังสือยังเป็นสีแดงได้
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(display, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
    cv2.putText(display, f"{label} ({dist:.1f})", (X1, Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Recognition (KNN)", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()