from __future__ import annotations

import cv2


def list_cameras(limit: int = 8) -> list[int]:
    """Quét nhanh camera OpenCV; có thể dùng iPhone virtual camera nếu Windows thấy nó."""
    found: list[int] = []
    for index in range(limit):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(index)
        cap.release()
    return found
