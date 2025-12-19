# memorymap/modules/utils.py
import cv2

def resize_for_yolo(frame, max_dim=640):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame
