# Gợi ý nhạc Việt - gói bàn giao làm app

Hướng dẫn phạm vi, giao diện, nghiệm thu và prompt để tiếp tục xây web app nằm
trong [`APP_BUILD_HANDOFF.md`](./APP_BUILD_HANDOFF.md).

## Chạy thử

Cần Python 3.14 và các gói trong `app-requirements.txt`.
Mở PowerShell ở thư mục vừa giải nén:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r app-requirements.txt
python -m ials_music_recommendation.main --top-k 10
```

Lệnh trên hiển thị danh sách phổ biến. Để gợi ý từ một hoặc nhiều bài người
dùng thích, truyền `track_id` (có thể lặp lại tham số):

```powershell
python -m ials_music_recommendation.main --track-id TRACK_ID --top-k 10
```

Backend có thể nạp model một lần khi khởi động:

```python
from ials_music_recommendation.model_bundle import TrainedModelBundle

models = TrainedModelBundle.from_directory()
recommendations = models.recommend_new_user(
    liked_track_ids=["TRACK_ID"],
    top_k=10,
)
```

Mỗi kết quả có `track_id`, `title`, `artist`, `score`; có thể kèm `genre` và
`dataset_source`. Hãy lưu `track_id` làm ID bài hát trong app. `score` là điểm
xếp hạng, không phải xác suất.

## Gợi ý emotion-aware cho demo CLI

CLI có chế độ emotion-aware tùy chọn, dùng emotion weight 0.5. Mặc định CLI và
API cũ vẫn giữ Content-Based/popularity để không đổi hành vi hiện tại. Full
emotion artifact hiện có vector cho 5.459 track trong catalog. Có 2.600 track
có lời đã duyệt và summary cảm xúc; các track còn lại chỉ có vector metadata,
không có đặc trưng cảm xúc từ lyrics. Track ngoài artifact sẽ báo lỗi thay vì
âm thầm chuyển sang model khác.

Pipeline dùng Ollama chạy cục bộ để tóm tắt cảm xúc cho lời đã được duyệt, rồi
tạo embedding kết hợp với tên bài, ca sĩ và thể loại cho recommender, cùng
embedding riêng chỉ từ tóm tắt cảm xúc để đo cosine trong bước đánh giá. Lyrics
không được gửi lên API bên ngoài. Đây là thử nghiệm lấy cảm hứng từ hướng
emotion-aware, không phải tái hiện chính xác paper LEMON và chưa chứng minh
chất lượng tốt hơn baseline.

Cài Ollama, tải hai model một lần và chạy từ thư mục cha chứa
`ials_music_recommendation`:

```powershell
ollama pull qwen2.5:3b
ollama pull bge-m3
python -m ials_music_recommendation.emotion_pipeline --max-items 335 --dataset-source dataViet
python -m ials_music_recommendation.emotion_pipeline
```

Pipeline chạy inference trên CPU, xử lý các lời đã duyệt, bỏ qua nội dung trùng
lặp và lưu checkpoint để có thể chạy lại sau khi bị gián đoạn. Kết quả được lưu
riêng trong `ials_music_recommendation/emotion_artifacts/`; không thay thế model
trong `trained_artifacts/`. Sau khi tạo đủ artifact, recommender thử nghiệm có
thể được khởi tạo trực tiếp bằng:

```python
from ials_music_recommendation.data_generator import load_song_catalog
from ials_music_recommendation.models.emotion_aware import EmotionAwareRecommender

songs = load_song_catalog()
model = EmotionAwareRecommender.from_artifacts(
    songs, "ials_music_recommendation/emotion_artifacts"
)
results = model.recommend_similar(track_id="TRACK_ID", top_k=10)
```

Khi artifacts đã có sẵn, chạy demo emotion-aware bằng track ID trong catalog:

```powershell
python -m ials_music_recommendation.main --track-id TRACK_ID --emotion-aware --top-k 10
```

Từ backend, có thể gọi trực tiếp:

```python
recommendations = models.recommend_emotion_aware(
    track_ids=["TRACK_ID"],
    top_k=10,
    emotion_weight=0.5,
)
```

Kết quả bao gồm `content_score`, `emotion_score` và `emotion_weight` ngoài các
trường bài hát thông thường. Điểm là điểm xếp hạng, không phải xác suất.

Đây mới là phép so sánh định tính/khởi đầu: tương tác người dùng hiện vẫn là dữ
liệu mô phỏng và chưa có nhãn cảm xúc chuẩn để kết luận mô hình tốt hơn.
Để xử lý toàn bộ lời đã duyệt thuộc nguồn `dataViet` và so sánh các trọng số:

```powershell
python -m ials_music_recommendation.emotion_pipeline --max-items 335 --dataset-source dataViet
python -m ials_music_recommendation.emotion_pipeline --max-items 335 --dataset-source dataViet --repair-truncated
python -m ials_music_recommendation.emotion_experiment
```

Catalog có 335 bài dataViet với lời đã duyệt trong tập thí nghiệm trước. Pipeline
lưu checkpoint để chạy tiếp sau khi gián đoạn và có thể tạo artifacts riêng
(`*.sample.*`) cho sample. Full artifact `manifest.json`/`song_ids.json` bao phủ
5.459 track hiện tại, với 2.600 summary từ lyrics đã duyệt. Tham số
`--repair-truncated` chạy lại các tóm tắt cũ bị cắt ở giới hạn từ, rồi cập nhật
embedding tương ứng; chạy bước này sau khi đã có summary checkpoint cũ.

Thử nghiệm so sánh emotion weight 0, 0.25, 0.5, 0.75 và 1 trên cùng 335 seed,
đo ILD@10, mức trùng top-10 với baseline và số nghệ sĩ khác nhau. LongTailShare
chỉ được tính nếu có dữ liệu popularity biến thiên; nếu không, báo cáo để trống
thay vì gán tất cả bài vào long tail. Thử nghiệm không đo relevance/accuracy vì
chưa có nhãn chuẩn. Kết quả và phiếu
đánh giá nằm trong `emotion_artifacts/experiment/`:

- `weight_comparison.csv`: tổng hợp theo trọng số.
- `weight_comparison_by_seed.csv`: kết quả từng seed.
- `summary_quality_review.csv`: phiếu chấm thủ công 30 tóm tắt, phân tầng theo
  thể loại, cộng thêm mọi tóm tắt placeholder trung tính nếu có.
- `blind_review_lists.csv`: 100 danh sách đề xuất đã gán mã ẩn danh để nhóm chấm
  mức hợp tâm trạng, phù hợp và đa dạng.
- `blind_review_key.csv`: bảng giải mã trọng số; chỉ mở sau khi nhóm chấm xong.

Để so sánh weight trên **toàn bộ catalog hiện tại** (5.459 seed), chạy:

```powershell
python -m ials_music_recommendation.emotion_experiment --scope all
```

Kết quả full catalog được lưu riêng trong
`emotion_artifacts/experiment/full_catalog/`, không ghi đè các bảng và báo cáo
thử nghiệm 335 bài dataViet ở trên. Bảng này tính lại ILD@10, độ trùng top-10
với baseline và các thống kê embedding trên mọi seed; summary tách riêng nhóm
seed có lời đã duyệt để tránh trộn lẫn với 2.859 track chỉ có metadata vector.

Nếu LongTailShare dùng lịch sử nghe mô phỏng đi kèm dự án, nó không phản ánh độ
phổ biến của người nghe thật. Nên dùng phiếu chấm tay hoặc local-metric
evaluator, rồi nhờ thành viên nhóm kiểm tra một phần trước khi chọn trọng số
hoặc đưa phương pháp này vào app.

### Chấm danh sách bằng metric local (miễn phí)

Thử nghiệm đầu tiên dùng Qwen 2.5 3B để tự chấm các danh sách; pilot cho cùng
điểm và nhận xét chung cho nhiều danh sách, nên không dùng các điểm đó để kết
luận. Evaluator hiện tại không hỏi LLM chấm điểm. Nó tính riêng:

- `mood_fit`: cosine BGE-M3 giữa vector chỉ tạo từ tóm tắt cảm xúc.
- `relevance`: điểm TF-IDF content similarity của recommender baseline.
- `variety`: ILD@10 từ khoảng cách cosine giữa các vector nội dung bài hát.

Các điểm 1–5 là **thứ hạng tương đối giữa năm danh sách của cùng một seed**;
điểm 3 có nghĩa là các danh sách hòa nhau. Đây không phải điểm tuyệt đối về
chất lượng, nhãn mood được kiểm chứng hay đánh giá của người nghe. Tóm tắt cảm
xúc chưa được xác thực thủ công vẫn có thể làm sai phép đo mood. Bảng score lưu
cả giá trị metric thô, độ đầy đủ metadata và confidence để kết quả có thể kiểm
tra được. Việc chấm dùng artifacts đã tạo cục bộ, không cần API key, không gọi
Ollama inference hay dịch vụ đám mây.

Chạy thử một seed từ thư mục cha dự án:

```powershell
python -m ials_music_recommendation.emotion_judge --max-seeds 1
```

Sau khi xem kết quả trong
`emotion_artifacts/experiment/local_metrics_scores.csv`, chạy cùng lệnh nhưng
bỏ `--max-seeds 1` để chấm tiếp toàn bộ 20 seed / 100 danh sách. Pipeline lưu
sau mỗi seed và chạy tiếp từ các điểm đã có.

Khi đã có đủ 100 dòng, lệnh tổng hợp sẽ giải mã và ghi
`local_metrics_summary_by_weight.csv`; nó từ chối nếu điểm chưa khớp toàn bộ
bảng khóa. Nên nhóm chấm thủ công một phần và xem lại các summary trước khi kết
luận weight nào có mood fit tốt hơn.
Hai bảng kết quả là `local_metrics_scores.csv` (từng danh sách và metric thô) và
`local_metrics_summary_by_weight.csv` (trung bình theo weight).
Phần kết quả và diễn giải sẵn để đưa vào báo cáo nằm trong
[`ket_qua_thu_nghiem_emotion_aware.md`](./emotion_artifacts/experiment/ket_qua_thu_nghiem_emotion_aware.md).

## File trong gói

- `ials_music_recommendation/main.py`: demo dòng lệnh.
- `ials_music_recommendation/model_bundle.py`: API nạp model và tạo gợi ý.
- `ials_music_recommendation/models/`: mã nguồn các model cần để nạp artifact.
- `ials_music_recommendation/trained_artifacts/`: model, catalog, mapping ID và
  manifest; cần giữ đồng bộ cả thư mục này.
- `app-requirements.txt`: phiên bản dependency tương ứng với artifacts.

Không đưa notebook, PDF, dữ liệu tương tác mô phỏng hoặc các file dataset thô
vào gói app. Ứng dụng hiện tại gợi ý từ bài người dùng thích và độ phổ biến
catalog; Funk SVD và Collaborative Filtering không được bật vì lịch sử hiện có
là dữ liệu mô phỏng, không phải hành vi người nghe thật. Để cá nhân hóa theo
lịch sử người dùng thật, app cần lưu tương tác thực và đánh giá lại mô hình.
