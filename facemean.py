import os
import re
import cv2
import numpy as np

# ===== ตั้งค่า =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_PATH = os.path.join(BASE_DIR, "kmeans_groups.jpg")
IMG_SIZE = (140, 180)      # ต้องเท่ากับใน FaceTrain_2.py
N_GROUPS = 5               # จำนวนกลุ่ม (ค่า k ของ k-means)
N_PCA = 20                 # ย่อข้อมูลเหลือกี่มิติก่อนจัดกลุ่ม (ช่วยลด noise)
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")


def preprocess(gray):
    """ขั้นตอนเดียวกับไฟล์เทรน: resize -> ปรับแสงให้เท่ากัน"""
    return cv2.equalizeHist(cv2.resize(gray, IMG_SIZE))


def load_dataset():
    """อ่านรูป รองรับทั้ง dataset/ชื่อ.jpg และ dataset/ชื่อ/รูป.jpg"""
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

    X, y = [], []
    for path, label in image_list:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ข้าม (อ่านไม่ได้): {path}")
            continue
        X.append(preprocess(img).flatten())
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def make_result_image(names, faces, groups):
    """สร้างภาพสรุป: 1 แถว = 1 กลุ่ม"""
    tw, th = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2      # ขนาดรูปย่อ
    cell_h = th + 22                                  # เผื่อที่เขียนชื่อใต้รูป
    rows = []
    for g in range(N_GROUPS):
        members = [i for i in range(len(names)) if groups[i] == g]
        head = np.full((cell_h, 110), 255, np.uint8)
        cv2.putText(head, f"Group {g + 1}", (8, cell_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        cells = [head]
        for i in members:
            cell = np.full((cell_h, tw + 6), 255, np.uint8)
            cell[0:th, 3:3 + tw] = cv2.resize(faces[i], (tw, th))
            cv2.putText(cell, names[i][:9], (3, cell_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cells.append(cell)
        rows.append(np.hstack(cells))

    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((cell_h, width - r.shape[1]), 255, np.uint8)]) for r in rows]
    sep = np.full((4, width), 180, np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, sep, r])
    return out


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ไม่พบโฟลเดอร์ dataset: {DATASET_DIR}")
        return

    print("กำลังอ่านรูปภาพจากโฟลเดอร์...")
    X, y = load_dataset()
    names = sorted(set(y.tolist()))

    if len(names) < N_GROUPS:
        print(f"มีแค่ {len(names)} คน ต้องมีอย่างน้อย {N_GROUPS} คนถึงจะแบ่ง {N_GROUPS} กลุ่มได้")
        return

    # 1 คน = 1 จุดข้อมูล (ถ้ามีหลายรูป ใช้หน้าเฉลี่ยของคนนั้น)
    feats = np.array([X[y == n].mean(axis=0) for n in names], dtype=np.float32)
    faces = [f.reshape(IMG_SIZE[1], IMG_SIZE[0]).astype(np.uint8) for f in feats]

    # PCA: ย่อจาก 25,200 พิกเซลเหลือไม่กี่มิติที่สำคัญ
    n_comp = max(1, min(N_PCA, len(names) - 1))
    mean, eigvec = cv2.PCACompute(feats, mean=None, maxComponents=n_comp)
    proj = cv2.PCAProject(feats, mean, eigvec).astype(np.float32)

    # K-means
    cv2.setRNGSeed(42)    # ให้ผลเหมือนเดิมทุกครั้งที่รัน
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, _ = cv2.kmeans(proj, N_GROUPS, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
    groups = labels.flatten()

    # ระยะห่างระหว่างทุกคู่ (ยิ่งน้อย = หน้ายิ่งเหมือน)
    dist = np.linalg.norm(proj[:, None, :] - proj[None, :, :], axis=2)

    print(f"\n===== ผลการแบ่ง {N_GROUPS} กลุ่ม =====")
    for g in range(N_GROUPS):
        members = [i for i in range(len(names)) if groups[i] == g]
        print(f"\nกลุ่ม {g + 1} ({len(members)} คน): " + ", ".join(names[i] for i in members))
        if len(members) >= 2:
            sub = dist[np.ix_(members, members)] + np.eye(len(members)) * 1e9
            a, b = np.unravel_index(np.argmin(sub), sub.shape)
            print(f"   คู่ที่หน้าเหมือนกันที่สุดในกลุ่ม: {names[members[a]]} กับ {names[members[b]]}")

    all_pairs = dist + np.eye(len(names)) * 1e9
    a, b = np.unravel_index(np.argmin(all_pairs), all_pairs.shape)
    print(f"\nคู่ที่หน้าเหมือนกันที่สุดทั้งห้อง: {names[a]} กับ {names[b]}")

    # ภาพสรุป
    result = make_result_image(names, faces, groups)
    ok, buf = cv2.imencode(".jpg", result)
    if ok:
        buf.tofile(RESULT_PATH)
        print(f"\nบันทึกภาพผลลัพธ์แล้ว: {RESULT_PATH}")

    cv2.imshow("K-means face groups (กดปุ่มใดก็ได้เพื่อปิด)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()