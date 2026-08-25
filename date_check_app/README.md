# Ứng dụng Kiểm tra Date

Ứng dụng này hoàn toàn tách khỏi app nhận diện vỉ.

Mẫu kiểm tra:

```text
NSX ddmmyy DM1
XX hh:mm
```

Mở bằng `python main.py`. Chọn camera hoặc video, rồi bấm **Vẽ ROI DATE trên khung hình**. ROI được lưu theo đúng nguồn camera/video nên chỉ cần chỉnh lại khi đổi góc hoặc vị trí camera.

`NSX mong đợi` và `Mã XX` là tùy chọn; để trống nếu chỉ kiểm tra định dạng. `DM1` được kiểm tra mặc định.
