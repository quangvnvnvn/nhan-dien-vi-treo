# Hệ thống nhận diện và đếm vỉ viên treo

Nền tảng desktop Python cho kiểm tra vỉ trên băng tải. Quyết định PASS chỉ được
đưa ra sau khi profile sản phẩm, slot, màu và confidence đều được xác thực.
Các trường hợp mơ hồ luôn trả về `UNKNOWN / NEED REVIEW`.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

## Kiểm thử nền tảng

```powershell
python -m unittest discover -s tests -v
```

## Trạng thái hiện tại

Đã có: cấu hình YAML, hồ sơ sản phẩm đa loại, SQLite audit trail, validation
slot/màu/confidence an toàn, TEST MODE ảnh tĩnh, và màn hình camera realtime
USB/RTSP. Camera capture và inference chạy trên các worker riêng; buffer chỉ
giữ frame mới nhất. Counter chỉ tăng sau khi Track ID vượt đường đếm đúng chiều
và có đủ số frame PASS liên tiếp. Ảnh FAIL/UNKNOWN tại đường đếm được lưu kèm
metadata JSON.

Chưa có: model segmentation đã train, hiệu chuẩn đầy đủ bằng ảnh/video thực,
tracker ByteTrack/BoT-SORT, huấn luyện YOLO, và các chỉ số accuracy thực tế.
Detector hình học hiện tại là bản an toàn cho giai đoạn hiệu chuẩn, không thể
cam kết nhận diện chính xác ở mọi góc/ánh sáng. Không dùng để quyết định sản
xuất trước khi có dataset nhãn và metric thực tế.

## Chạy camera realtime

1. Mở tab `CAMERA REALTIME`.
2. Chọn Product Profile đã xác nhận, chọn Camera USB (ví dụ `0`) hoặc RTSP.
3. Đặt ROI, đường đếm và chiều băng tải, rồi bấm `BẮT ĐẦU`.
4. Chỉ kết quả PASS xác thực đủ điều kiện mới được đếm. UNKNOWN/FAIL không
   được đếm và được lưu để review khi vượt đường đếm.
