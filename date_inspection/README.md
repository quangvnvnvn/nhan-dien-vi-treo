# Kiểm tra DATE trên băng tải

Phiên bản này hoàn thành Giai đoạn 1–2: một pipeline dùng chung cho camera và video, điều khiển video (play/pause/stop/restart/seek/frame-step/tốc độ), `PRODUCT DETECTION ZONE`, detector nền/contour nhẹ và state machine chống đếm trùng.

## Chạy

```powershell
cd date_inspection
python -m pip install -r requirements.txt
python main.py
```

1. Mở **TEST VIDEO** hoặc **GIÁM SÁT**.
2. Nạp nguồn frame.
3. Qua tab **PHÁT HIỆN SẢN PHẨM**, bấm *Vẽ PRODUCT DETECTION ZONE*, rồi kéo chuột trên ảnh.
4. Chỉnh các ngưỡng và bấm *Lưu và áp dụng detector*.

Video test và camera đều gọi cùng `InspectionPipeline`, `MotionContourDetector` và `ProductTracker`. OCR/DATE ROI/candidate sharpness sẽ được bổ sung ở Giai đoạn 3–4, nên trigger hiện chỉ tạo `Product ID` và debug trực tiếp.
