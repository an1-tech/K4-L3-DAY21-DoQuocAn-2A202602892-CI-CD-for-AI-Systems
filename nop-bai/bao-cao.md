# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Đỗ Quốc An |
| MSSV | 2A202602892 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/an1-tech/K4-L3-DAY21-DoQuocAn-2A202602892-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 đạt F1 cao nhất và vượt ngưỡng 0.65. Lần 1 có accuracy cao nhất nhưng F1 thấp hơn, cho thấy hai chỉ số xếp hạng khác nhau. Lần 2 dùng learning rate thấp, ít cây, cây nông, đạt F1 0.6051 nên không đủ chất lượng. Learning rate nhỏ thường cần nhiều cây hơn; tuy nhiên các thí nghiệm thay đổi nhiều tham số nên chưa tách riêng được ảnh hưởng. Các lần chạy dùng random_state=42 và cùng holdout 500 mẫu. Bộ tốt nhất được lưu vào params.yaml và chạy lại để lưu model.

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Dữ liệu Adult có khoảng 24,8% mẫu thu nhập trên 50K. Luôn đoán thu nhập thấp vẫn đạt accuracy khoảng 75,2%, nhưng F1 lớp dương bằng 0 vì bỏ sót mọi trường hợp thu nhập cao. F1 kết hợp precision và recall, phản ánh dự đoán dương sai và bỏ sót. Pipeline yêu cầu F1 lớp dương ít nhất 0.65; accuracy dùng tham khảo. Hàm f1_score dùng average="binary", pos_label=1, zero_division=0. Weighted F1 tổng hợp hai lớp theo số lượng mẫu; macro F1 lấy trung bình không trọng số giữa hai lớp. Cả hai khác F1 riêng của lớp dương mà rubric yêu cầu.

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow không khởi tạo được SQLite. | SQLAlchemy 2.1.3 thiếu thành phần MLflow 2.13 cần. | Cố định SQLAlchemy 2.0.30; kiểm tra thư viện và 24 tests đều đạt. |
| Tạo bucket bị SCP chặn tại hai region. | Policy hạn chế S3 ở us-east-1 và us-west-2. | Đọc policy từ Management Account; dùng Sydney ap-southeast-2 phù hợp chính sách. |
| Release thất bại khi copy code. | Fingerprint host key không khớp. | Dùng OpenSSH, xác minh và chọn ED25519; pipeline chạy thành công. |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** Trên cùng holdout, F1 tăng 0.0205 và accuracy tăng 0.0080; dữ liệu bổ sung cải thiện kết quả quan sát được nhưng không chứng minh thêm dữ liệu luôn tốt hơn. Commit chỉ cập nhật con trỏ DVC đã tự kích hoạt bốn jobs và triển khai lại API trên EC2.

Nguồn số liệu: [Bước 2](https://github.com/an1-tech/K4-L3-DAY21-DoQuocAn-2A202602892-CI-CD-for-AI-Systems/actions/runs/37608044769/job/112748686507#step:9:32), [Bước 3](https://github.com/an1-tech/K4-L3-DAY21-DoQuocAn-2A202602892-CI-CD-for-AI-Systems/actions/runs/37608619769/job/112750564486#step:9:32).
