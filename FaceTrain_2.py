import os
import re
import cv2
import numpy as np

# ===== ตั้งค่า =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # โฟลเดอร์ที่ไฟล์นี้อยู่
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "knn_model.npz")
IMG_SIZE = (140, 180)        # (กว้าง, สูง) ให้เท่ากับกรอบในกล้อง 250-390 x 120-300
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")


def preprocess(gray):
    """ขั้นตอนเดียวกับตอนทำนาย: resize -> ปรับแสงให้เท่ากัน"""
    return cv2.equalizeHist(cv2.resize(gray, IMG_SIZE))


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ไม่พบโฟลเดอร์ dataset: {DATASET_DIR}")
        return

    print("กำลังอ่านรูปภาพจากโฟลเดอร์...")

    # รองรับ 2 แบบ:
    #   dataset/ชื่อ.jpg, dataset/ชื่อ_2.jpg, dataset/ชื่อ (3).jpg  -> ชื่อมาจากชื่อไฟล์
    #   dataset/ชื่อ/รูป.jpg                                     -> ชื่อมาจากชื่อโฟลเดอร์
    image_list = []
    for item in sorted(os.listdir(DATASET_DIR)):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            for file in sorted(os.listdir(item_path)):
                if file.lower().endswith(VALID_EXT):
                    image_list.append((os.path.join(item_path, file), item))
        elif item.lower().endswith(VALID_EXT):
            stem = os.path.splitext(item)[0]
            label = re.sub(r"[\s_\-]*\(?\d+\)?$", "", stem).strip() or stem
            image_list.append((item_path, label))

    X_train, y_train = [], []
    for path, label in image_list:
        data = np.fromfile(path, dtype=np.uint8)            # อ่าน path ภาษาไทย/ช่องว่างได้
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ข้าม (อ่านไม่ได้): {path}")
            continue
        X_train.append(preprocess(img).flatten())
        y_train.append(label)

    if not X_train:
        print("ไม่พบรูปภาพสำหรับเทรน")
        return

    for label in sorted(set(y_train)):
        print(f"  {label}: {y_train.count(label)} รูป")

    X = np.array(X_train, dtype=np.uint8)
    y = np.array(y_train)
    np.savez(MODEL_PATH, X=X, y=y, img_size=np.array(IMG_SIZE))

    print(f"บันทึกโมเดลแล้ว: {MODEL_PATH}")
    print(f"จำนวนข้อมูล: {X.shape[0]} รูป, ขนาดต่อรูป: {X.shape[1]} ค่า")


if __name__ == "__main__":
    main()