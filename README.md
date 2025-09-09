# genai-network-optimization

## Dataset
- **Nguồn**: 5G Resource Allocation Dataset📡:Optimizing Band Dataset từ Kaggle.
- **Mô tả**:Dataset chứa thông tin phân bổ tài nguyên 5G theo loại ứng dụng, gồm 400 dòng dữ liệu. Các cột chính:

Timestamp: Thời điểm ghi nhận dữ liệu.

User_ID: Mã người dùng.

Application_Type: Loại ứng dụng (Video Call, Streaming, IoT, …).

Signal_Strength: Cường độ tín hiệu (dBm).

Latency: Độ trễ mạng (ms).

Required_Bandwidth: Băng thông yêu cầu (Kbps/Mbps).

Allocated_Bandwidth: Băng thông được cấp phát (Kbps/Mbps).

Resource_Allocation: Tỷ lệ phân bổ tài nguyên (%).
- **Synthetic Data**: Được tạo bằng GAN để tăng số lượng mẫu và mô phỏng các kịch bản mạng bất định, lưu trong `data/synthetic/synthetic_data.csv`.

## Tiến độ
- [x] Thiết lập cấu trúc project
- [x] Khám phá dữ liệu 
- [x] Tiền xử lý dữ liệu (xóa cột, chuyển kiểu ,chia train/test, cân bằng dữ liệu dựa trên target (RA) )
- [ ] Data Augmentation bằng GAN