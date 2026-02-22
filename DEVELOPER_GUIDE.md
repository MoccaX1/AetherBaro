# Hướng dẫn Quy trình Phát triển (Workflow Guide)

Tài liệu này hướng dẫn cách sử dụng hệ thống quản lý phiên bản và lịch sử thay đổi tự động cho dự án **AetherBaro**.

## 1. Quy trình làm việc hàng ngày (Daily Workflow)

Khi bạn thực hiện các thay đổi code, hãy sử dụng Git để commit một cách chi tiết. Mỗi tin nhắn commit sẽ là một dòng trong Changelog sau này.

**Cú pháp khuyến nghị:**
```bash
git add .
git commit -m "Loại: Mô tả ngắn gọn về thay đổi"
```

*Ví dụ:*
- `git commit -m "feat: Thêm biểu đồ phổ năng lượng FFT ở Layer 2"`
- `git commit -m "fix: Sửa lỗi nhãn Pmin bị đè lên trục thời gian"`
- `git commit -m "chore: Cập nhật tài liệu hướng dẫn cài đặt"`

## 2. Quy trình Phát hành Phiên bản (Release Workflow)

Khi bạn đã sẵn sàng chốt một phiên bản mới, thay vì tự sửa file `CHANGELOG.md`, hãy sử dụng script `version_up.py`.

### Bước 1: Quyết định mức độ thay đổi
- **patch**: Sửa lỗi nhỏ, không thêm tính năng (v1.0.0 -> v1.0.1)
- **minor**: Thêm tính năng mới nhưng vẫn tương thích cũ (v1.0.0 -> v1.1.0)
- **major**: Thay đổi lớn, phá vỡ cấu trúc cũ (v1.0.0 -> v2.0.0)

### Bước 2: Chạy lệnh tự động
Mở terminal tại thư mục gốc của dự án và chạy:

```bash
python version_up.py [mức_độ]
```

*Ví dụ:* `python version_up.py patch`

### Bước 3: Kết quả tự động
Sau khi chạy lệnh, hệ thống sẽ tự động thực hiện:
1.  **Quét Git History**: Lấy toàn bộ các commit message kể từ phiên bản trước.
2.  **Cập nhật `VERSION`**: Tăng số hiệu phiên bản.
3.  **Cập nhật `CHANGELOG.md`**: Chèn thông tin phiên bản mới, ngày tháng và danh sách commit kèm link mã hash.
4.  **Git Commit Release**: Tạo một commit mới với nội dung `chore(release): X.Y.Z`.
5.  **Git Tagging**: Tạo một tag `vX.Y.Z` để đánh dấu cột mốc trên Git.

## 3. Lưu ý quan trọng
- Luôn đảm bảo bạn đã `git commit` hết các thay đổi code trước khi chạy `version_up.py`.
- Script này yêu cầu Python và Git đã được cài đặt và cấu hình trong PATH của Windows.
