# Kết quả thử nghiệm Emotion-aware

## Bổ sung: so sánh weight trên toàn catalog

Sau khi tạo full emotion artifact, đã chạy lại so sánh năm weight trên toàn bộ
5.459 track hợp lệ trong catalog; mỗi track lần lượt được dùng làm seed. Có
2.600 seed có lời đã duyệt và summary cảm xúc, còn 2.859 seed không có lời đã
duyệt. Các vector kết hợp dùng metadata cho toàn bộ catalog và summary lời bài
hát khi có; vì thế các metric dưới đây mô tả xếp hạng từ embedding kết hợp,
không phải phép đo cảm xúc thuần túy trên mọi track.

| Emotion weight | ILD@10 toàn catalog | Trùng top-10 với baseline | Điểm embedding trung bình (0–1) | Nghệ sĩ trung bình/top-10 | Top-10 có lời đã duyệt |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.6813 | 100.0% | 0.8378 | 5.48 | 48.2% |
| 0.25 | 0.6843 | 88.4% | 0.8434 | 5.27 | 48.3% |
| **0.50** | **0.6909** | **74.5%** | **0.8501** | **5.04** | **48.5%** |
| 0.75 | 0.7074 | 57.8% | 0.8572 | 4.91 | 49.6% |
| 1.00 | 0.7743 | 33.3% | 0.8630 | 5.38 | 53.8% |

Ở weight 0.5, ILD@10 tăng khoảng 0.0096 so với baseline; top-10 trùng baseline
trung bình 74.5%. Khi chỉ xét 2.600 seed có summary lời đã duyệt, weight 0.5
có ILD@10 trung bình 0.6917 và top-10 trùng baseline 66.8%. Đây là phân nhóm
theo việc seed có lyrics, nhưng các bài được đề xuất vẫn có thể thuộc phần
metadata-only.

Kết quả toàn catalog cho thấy weight cao hơn làm thay đổi danh sách và tăng
ILD/điểm embedding, nhưng **không chứng minh gợi ý hợp tâm trạng hoặc được
người nghe ưa thích hơn**. 2.859 track không có summary từ lyrics; embedding của
chúng chỉ thể hiện metadata. 13 summary dùng nội dung dự phòng trung tính và
cần được xem xét khi đánh giá cảm xúc. Do chưa có nhãn chuẩn hay phản hồi người
dùng, weight 0.5 tiếp tục là cấu hình demo cân bằng được đề xuất, không phải
weight tối ưu đã được xác nhận.

Kết quả đầy đủ theo từng seed nằm trong `full_catalog/weight_comparison_by_seed.csv`;
bảng tổng hợp nằm trong `full_catalog/weight_comparison.csv`. Manifest ghi rõ
phạm vi và số lượng seed trong `full_catalog/full_catalog_experiment_manifest.json`.

## Mục tiêu

Thử nghiệm khảo sát ảnh hưởng của trọng số cảm xúc trong hệ gợi ý âm nhạc
Emotion-aware. Content-Based hiện tại được dùng làm mốc so sánh; emotion weight
được thay đổi lần lượt qua 0, 0.25, 0.5, 0.75 và 1.0.

## Thiết lập

Catalog thử nghiệm gồm 335 bài dataViet có lời đã được duyệt, thuộc năm thể
loại. Với mỗi bài làm seed, hệ thống tạo danh sách top-10 ở từng trọng số.
Điểm kết hợp của hệ thống là:

```text
score = (1 - emotion_weight) * content_score
      + emotion_weight * emotion_score
```

`content_score` là độ tương đồng Content-Based dựa trên TF-IDF và metadata;
`emotion_score` là độ tương đồng embedding giữa biểu diễn cảm xúc của seed và
bài ứng viên. Thử nghiệm tính các metric top-10 trên cùng tập seed cho cả năm
trọng số.

Ngoài so sánh toàn bộ 335 seed, 20 seed được chọn phân tầng theo thể loại để
tạo 100 danh sách chấm mù (năm trọng số cho mỗi seed). Các metric `mood fit`,
`relevance` và `variety` của tập này được chuyển thành thứ hạng tương đối từ
1–5 giữa năm danh sách của cùng seed; nếu hòa, điểm là 3. Các điểm sau đó được
tổng hợp trên 20 seed. Đây là thứ hạng metric, không phải đánh giá trực tiếp
của người nghe.

## Kết quả trên 335 seed

| Emotion weight | ILD@10 | Trùng top-10 với baseline | Cosine cảm xúc trung bình | Số nghệ sĩ trung bình/top-10 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.6638 | 100.0% | 0.8539 | 8.02 |
| 0.25 | 0.6683 | 91.2% | 0.8572 | 7.97 |
| **0.50** | **0.6780** | **77.6%** | **0.8622** | **7.98** |
| 0.75 | 0.6984 | 58.4% | 0.8683 | 7.98 |
| 1.00 | 0.7706 | 29.4% | 0.8738 | 8.06 |

So với baseline ở weight 0, cấu hình 0.5 làm ILD@10 tăng khoảng 0.0142 điểm
(0.6638 → 0.6780), trong khi trung bình 77.6% bài trong top-10 vẫn trùng
baseline. Khi weight tăng lên 0.75 và 1.0, danh sách đa dạng hơn theo ILD,
nhưng thứ hạng thay đổi nhiều hơn so với baseline. LongTailShare không được
báo cáo vì tập dataViet không có tín hiệu popularity biến thiên đáng tin cậy.

## Kết quả thứ hạng trên 20 seed chấm mù

| Emotion weight | Mood fit | Relevance | Variety | Độ đầy đủ metadata nghệ sĩ |
|---:|---:|---:|---:|---:|
| 0.00 (baseline) | 2.00 | 4.90 | 1.60 | 92.5% |
| 0.25 | 2.75 | 4.25 | 1.90 | 93.5% |
| **0.50** | **3.15** | **3.10** | **2.95** | **90.0%** |
| 0.75 | 3.45 | 2.05 | 3.80 | 87.0% |
| 1.00 | 4.05 | 1.00 | 5.00 | 80.0% |

Các thứ hạng cho thấy trade-off trong mẫu chấm: tăng trọng số cảm xúc đi cùng
với mood-fit và variety cao hơn, nhưng relevance thấp hơn. Vì weight 0.5 nằm
gần giữa thang và giữ được một phần lớn danh sách của baseline, nhóm chọn
**emotion weight = 0.5 làm cấu hình cân bằng cho bản thử nghiệm/demo**, thay vì
khẳng định đây là weight tối ưu.

## Giới hạn và cách diễn giải

- Tập chấm mù có 20 seed, không phải đánh giá người dùng thật. Bảng 1–5 là thứ
  hạng tương đối trên năm danh sách của cùng seed, không có ý nghĩa như điểm
  chất lượng tuyệt đối.
- Mood fit dùng cosine BGE-M3 giữa các embedding chỉ tạo từ tóm tắt cảm xúc.
  Đây là phép đo tương đồng ngữ nghĩa, không phải nhãn cảm xúc đã được xác nhận.
  Một số tóm tắt cần được kiểm tra thủ công trước khi diễn giải điểm mood.
- Relevance trong bảng chấm mù là độ tương đồng Content-Based trung bình; ILD
  đo độ khác nhau về biểu diễn nội dung. Cả hai là metric thay thế, không chứng
  minh trực tiếp người nghe thích hoặc chọn bài được gợi ý.
- Content score và emotion score là hai đại lượng có phân phối khác nhau; phép
  trộn tuyến tính hiện tại chưa được hiệu chuẩn để làm chúng cùng thang đo.
  Vì vậy không diễn giải weight như tỷ lệ đóng góp chính xác về mặt cảm nhận.
- Không có nhãn cảm xúc chuẩn hoặc lịch sử nghe thật cho subset này. Không thể
  kết luận Emotion-aware có accuracy cao hơn Content-Based.

**Kết luận báo cáo:** Trong thử nghiệm trên 335 bài dataViet, tăng emotion
weight làm ILD@10 và độ tương đồng embedding cảm xúc tăng, đồng thời làm danh
sách thay đổi nhiều hơn so với Content-Based baseline. Trên mẫu chấm mù 20
seed, weight 0.5 cho thứ hạng trung gian giữa mood-fit, relevance và variety;
do đó được chọn làm cấu hình cân bằng cho demo. Kết quả mang tính thăm dò và
cần được đối chiếu với đánh giá người nghe trước khi khẳng định mức độ phù hợp.

## Tệp kết quả

- `weight_comparison.csv`: metric tổng hợp trên 335 seed.
- `weight_comparison_by_seed.csv`: metric theo từng seed và weight.
- `local_metrics_scores.csv`: metric thô và thứ hạng trên 100 danh sách chấm mù.
- `local_metrics_summary_by_weight.csv`: trung bình các metric trên 20 seed chấm mù.
- `blind_review_lists.csv`: các danh sách chấm mù.
- `blind_review_key.csv`: bảng giải mã weight; chỉ dùng sau khi khóa đánh giá.
- `summary_quality_review.csv`: phiếu kiểm tra summary cảm xúc.
