# Bàn giao xây dựng app V-MusicRec

Tài liệu này mô tả phạm vi app demo cần làm tiếp theo. Có thể gửi nguyên phần
"Prompt giao việc" ở cuối cho trợ lý lập trình.

Gói ZIP `music_recommender_app_handoff.zip` ở thư mục đồ án chứa backend,
artifacts đã tạo và tài liệu bàn giao cần cho app. ZIP không chứa thư mục
Dataset, notebook hoặc PDF bài giảng; các tài liệu đó vẫn được giữ nguyên trong
project gốc.

## 1. Kết luận đánh giá hiện tại

Đề xuất dùng Emotion-aware với `emotion_weight = 0.5` làm cấu hình demo mặc
định. Đây là lựa chọn cân bằng để trình diễn, không phải trọng số tối ưu đã
được chứng minh.

- Thử nghiệm so sánh weight trước đó dùng 335 seed dataViet: ILD@10 tăng từ
  `0.6638` ở baseline lên `0.6780` ở weight 0.5; trung bình 77.6% bài trong
  top-10 vẫn trùng baseline.
- So sánh mới trên toàn bộ 5.459 track cho ILD@10 `0.6813` ở baseline và
  `0.6909` ở weight 0.5; top-10 trùng baseline trung bình 74.5%. Trong 2.600
  seed có lời đã duyệt, weight 0.5 có ILD@10 `0.6917` và overlap 66.8%.
- Trên 20 seed đánh giá mù bằng metric local, weight 0.5 có thứ hạng trung
  bình `mood fit = 3.15`, `relevance = 3.10`, `variety = 2.95` trên thang thứ
  hạng 1–5 giữa năm trọng số.
- Weight 1.0 có mood fit và variety cao hơn, nhưng relevance thấp nhất. Baseline
  vẫn có relevance cao nhất trong thử nghiệm.
- Đây không phải điểm người nghe chấm: chưa có nhãn cảm xúc chuẩn, lịch sử nghe
  thật hoặc thử nghiệm người dùng. Không được quảng bá kết quả là accuracy hay
  bằng chứng người dùng thích gợi ý hơn.

Vì vậy app nên trình bày 0.5 như một cấu hình thử nghiệm cân bằng và có lựa
chọn so sánh với Content-Based, không gọi là "AI hiểu chính xác tâm trạng".

## 2. Phạm vi và giới hạn đã xác nhận

- Full emotion artifact hiện bao phủ **5.459 track hợp lệ** trong catalog.
  Trong đó có **2.600 track với lyrics được duyệt và tóm tắt cảm xúc**; các bài
  còn lại chỉ có vector tạo từ metadata (tên bài, nghệ sĩ, thể loại), không có
  tín hiệu cảm xúc lấy từ lời bài hát. Catalog bỏ một hàng không có ID/title.
- App dùng `emotion_artifacts/song_ids.json` làm danh sách ID hỗ trợ. Không dùng
  `song_ids.sample.json` để lọc UI nếu full artifact có sẵn.
- Cho phép chọn bài trong toàn catalog khi dùng Emotion-aware, nhưng nếu seed
  không có summary cảm xúc đã duyệt thì hiển thị ghi chú phù hợp; không khẳng
  định nó được phân tích cảm xúc từ lyrics.
- Embedding và tóm tắt đã tạo sẵn. **Không cần Ollama, API key hay gọi dịch vụ
  bên ngoài khi chạy app.** Không chạy lại pipeline trong luồng khởi động app.
- Không có URL phát nhạc được xác nhận trong contract hiện tại. Không tự tạo
  nút nghe hoặc URL giả.
- Dữ liệu tương tác đi kèm dự án là dữ liệu mô phỏng; không dùng để tuyên bố
  chất lượng cá nhân hóa hoặc mức phổ biến thực tế.
- Tham khảo các metric và giới hạn trong
  [`emotion_artifacts/experiment/ket_qua_thu_nghiem_emotion_aware.md`](./emotion_artifacts/experiment/ket_qua_thu_nghiem_emotion_aware.md).
- Kết quả mới theo từng seed và weight ở
  `emotion_artifacts/experiment/full_catalog/`; không trộn với bảng thử nghiệm
  335 seed dataViet trước đó.

## 3. Trải nghiệm app đề xuất

Làm một web app demo nhỏ bằng **Streamlit** để nhóm dễ cài, chạy và trình bày.
Giữ giao diện tiếng Việt, responsive ở mức cơ bản, không cần đăng nhập/database.

### Màn hình chính

1. Tiêu đề: **V-MusicRec — Gợi ý nhạc Việt theo nội dung và cảm xúc**.
2. Một dòng giải thích: "Bản demo bao phủ catalog 5.459 bài; 2.600 bài có lời
   đã duyệt để tạo tóm tắt cảm xúc. Kết quả chưa được xác nhận theo sở thích
   người nghe."
3. Tìm và chọn **1–3 bài seed** trong catalog theo tên bài hoặc nghệ sĩ. Dùng
   `track_id` làm giá trị định danh, không dùng title làm ID.
4. Chọn chế độ:
   - **So sánh hai phương pháp** (mặc định): Content-Based và Emotion-aware
     (weight 0.5).
   - Content-Based.
   - Emotion-aware (weight 0.5).
5. Nút **Tạo gợi ý**; mặc định top 10, có thể cho chọn 5/10/15.
6. Hiển thị kết quả thành hai cột khi so sánh. Mỗi bài hiển thị thứ hạng, tên
   bài, nghệ sĩ, thể loại nếu có. Có thể hiển thị `content_score` và
   `emotion_score` như điểm nội bộ, kèm lời nhắc rằng đó không phải xác suất.
7. Không hiển thị lyrics đầy đủ. Không tự sinh nhãn cảm xúc bằng cách diễn
   giải score. Nếu hiển thị summary có sẵn, phải ghi rõ đó là summary máy tạo
   và không phải nhãn đã được kiểm chứng.
8. Thêm ghi chú giới hạn dữ liệu và phương pháp ở cuối trang; không thêm tùy
   chọn weight tùy ý trong bản đầu.

### Hành vi lỗi

- Chưa chọn seed: yêu cầu người dùng chọn ít nhất một bài.
- Seed không có summary lyrics đã duyệt: cho phép gợi ý bằng embedding metadata
  nhưng giải thích rằng bài đó không có tín hiệu cảm xúc từ lyrics.
- Thiếu artifact/model: báo đường dẫn/tên artifact bị thiếu và hướng dẫn kiểm
  tra gói dữ liệu; không chạy pipeline tự động, không trả danh sách giả.
- Bài seed không có kết quả: báo rõ, không hiển thị success rỗng.
- Không lưu lịch sử tìm kiếm hoặc feedback của người dùng trong bản đầu.

## 4. Backend hiện có để tái sử dụng

Backend đã có `TrainedModelBundle` và hàm gọi Emotion-aware:

```python
from ials_music_recommendation.model_bundle import TrainedModelBundle

models = TrainedModelBundle.from_directory()

baseline = models.recommend_similar(
    track_ids=["TRACK_ID"],
    top_k=10,
)

emotion = models.recommend_emotion_aware(
    track_ids=["TRACK_ID"],
    top_k=10,
    emotion_weight=0.5,
)
```

Kết quả gồm `track_id`, `title`, `artist`, `score`; tùy track có thể có
`genre`, `genre_confidence`, `dataset_source`. Emotion-aware có thêm
`content_score`, `emotion_score`, `emotion_weight`. Khởi tạo `models` một lần
khi app chạy, không nạp lại artifacts mỗi lần bấm nút.

Danh sách lựa chọn phải đối chiếu ID trong:

- `ials_music_recommendation/emotion_artifacts/song_ids.json`
- Metadata bài hát trong `ials_music_recommendation/trained_artifacts/catalog.csv`
- Tóm tắt đã duyệt trong `ials_music_recommendation/emotion_artifacts/emotion_summaries.csv`

Hàm `recommend_emotion_aware` kiểm tra seed thuộc emotion artifacts. Full
artifact có vector cho toàn bộ catalog hiện tại. Nếu UI có cho loại trừ bài đã
chọn thì truyền ID hợp lệ; seed tự được loại khỏi kết quả.

## 5. Cài đặt và chạy app sau khi được tạo

Hiện tại CLI emotion-aware đã chạy được; đây là lệnh smoke test từ thư mục cha
có `ials_music_recommendation`:

```powershell
Set-Location "D:\Học tập\DS300\Đồ án"
$trackId = (Get-Content "ials_music_recommendation\emotion_artifacts\song_ids.json" -Raw | ConvertFrom-Json)[0]
python -m ials_music_recommendation.main --track-id $trackId --emotion-aware --top-k 5
```

Khi app Streamlit được tạo, bổ sung `streamlit` vào
`ials_music_recommendation/app-requirements.txt`, sau đó cài và chạy:

```powershell
Set-Location "D:\Học tập\DS300\Đồ án"
python -m pip install -r ials_music_recommendation\app-requirements.txt
python -m streamlit run ials_music_recommendation\app.py
```

Không cài lại Ollama hoặc model Qwen/BGE chỉ để chạy app. Các artifacts hiện
tại đã được tạo offline.

## 6. Checklist nghiệm thu

- [ ] App khởi động theo hướng dẫn, không cần API key hoặc Ollama đang chạy.
- [ ] Có thể tìm/chọn seed bằng tên bài/nghệ sĩ; xử lý đúng Unicode tiếng Việt.
- [ ] Chỉ cho chọn seed có emotion artifact trong chế độ Emotion-aware.
- [ ] So sánh dùng đúng cùng seed và top-k cho hai phương pháp.
- [ ] Emotion-aware gọi backend `recommend_emotion_aware(..., emotion_weight=0.5)`.
- [ ] Seed không xuất hiện trong danh sách gợi ý.
- [ ] Bài gợi ý có `track_id` duy nhất và metadata hiển thị đúng.
- [ ] Hiển thị lỗi dễ hiểu khi seed không hỗ trợ hoặc artifacts thiếu.
- [ ] Không nhầm score với xác suất; không tuyên bố độ chính xác đã được xác nhận.
- [ ] Chạy thử baseline và Emotion-aware với ít nhất một seed hợp lệ.
- [ ] README cập nhật lệnh cài, chạy, giới hạn và ảnh chụp giao diện nếu có.

## 7. Prompt giao việc

> Hãy xây dựng web app demo Streamlit bằng tiếng Việt cho đồ án V-MusicRec
> ngay trong thư mục `ials_music_recommendation`, theo tài liệu
> `APP_BUILD_HANDOFF.md`. Trước khi sửa, đọc `APP_HANDOFF_README.md`,
> `model_bundle.py`, `models/emotion_aware.py` và kiểm tra artifacts hiện có.
> Tái sử dụng `TrainedModelBundle`; không viết lại thuật toán gợi ý, không tạo
> dữ liệu giả, không cần Ollama/API key lúc chạy. Cho người dùng tìm và chọn
> 1–3 seed trong catalog 5.459 bài; hỗ trợ so sánh Content-Based với
> Emotion-aware weight 0.5 và hiển thị metadata/điểm một cách trung thực. Nêu
> rõ chỉ 2.600 track có summary cảm xúc từ lyrics đã duyệt; các track còn lại
> chỉ dùng vector metadata. Không gọi điểm là xác suất/accuracy. Cập nhật
> `app-requirements.txt` và hướng dẫn chạy, xử lý lỗi rõ ràng, rồi tự chạy
> smoke test cho cả baseline và Emotion-aware. Chỉ kết thúc khi app chạy được
> theo hướng dẫn và các mục checklist nghiệm thu quan trọng đã được kiểm tra.
