import os
import cv2
import numpy as np


def knn_similarity(X, y, z, k=5, threshold=3000.0):
    """
    คำนวณระยะห่าง (Euclidean Distance) หาเพื่อนบ้านใกล้ที่สุด k คน
    threshold: ค่าระยะห่างสูงสุด ถ้ายิ่งน้อยยิ่งเหมือนกันมาก (ปรับเพิ่ม/ลดได้ตามความเหมาะสม)
    """
    # คำนวณ Euclidean Distance กับรูปทั้งหมดในคลัง
    distances = np.sqrt(np.sum((X - z) ** 2, axis=1))

    # ดึงดรรชนีของรูปที่ใกล้ที่สุด k อันดับแรก
    idx = np.argsort(distances)[:k]

    # เก็บผลลัพธ์คนที่ผ่านเกณฑ์ Threshold (ระยะห่างน้อยกว่าค่าที่กำหนด)
    matched_candidates = []
    for i in idx:
        if distances[i] <= threshold:
            matched_candidates.append(
                {"name": y[i], "distance": distances[i]}
            )

    # ดึงชื่อคนที่ผ่านเกณฑ์แบบไม่ซ้ำกัน
    unique_names = list(set([item["name"] for item in matched_candidates]))

    return unique_names, matched_candidates


# โหลดโมเดล
if not os.path.exists("knn_model.npz"):
    print("❌ ไม่พบไฟล์ knn_model.npz กรุณารัน train_knn.py ก่อน")
    exit()

data = np.load("knn_model.npz")
X_train, y_train = data["X"], data["y"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("เปิดกล้องไม่ได้")
        break

    # พิกัดกรอบตัดภาพ
    x1, y1, x2, y2 = 250, 120, 390, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size > 0:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        z = gray_face.flatten()

        if len(X_train) > 0 and z.shape[0] == X_train.shape[1]:
            # เรียกใช้ฟังก์ชันคำนวณความคล้ายคลึง
            # สามารถปรับ threshold เพิ่มขึ้นได้ถ้าอยากให้หาง่ายขึ้น
            matched_names, details = knn_similarity(
                X_train, y_train, z, k=5, threshold=3500.0
            )

            count = len(matched_names)

            # แสดงจำนวนคนที่คล้ายบนหัวกรอบ
            text_summary = f"Similar to: {count} person(s)"
            cv2.putText(
                frame,
                text_summary,
                (x1 - 20, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # ถ้าพบคนที่คล้าย ให้แสดงรายชื่อออกมา
            if count > 0:
                names_str = ", ".join(matched_names)
                cv2.putText(
                    frame,
                    f"Names: {names_str}",
                    (x1 - 20, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Unknown / Low similarity",
                    (x1 - 20, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow("KNN Multi-Face Matching", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()