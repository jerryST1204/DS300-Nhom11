# Báo cáo dataset và quá trình tiền xử lý

## 1. Mục tiêu và phạm vi

Dự án hiện có hai nguồn dữ liệu âm nhạc được xử lý riêng:

1. **Metadata từ Hugging Face**: lấy từ dataset `Toan-Minh-Duong-Son/vietnamese-music-dataset` bằng `CleanDataStream.py`. Nguồn này cung cấp metadata bài hát và phụ đề, không tải file âm thanh.
2. **Các đoạn âm thanh trong `Music/`**: quét và tạo metadata cấp đoạn/cấp bài bằng `cleanDataViet.py`. Nguồn này gồm các thư mục thể loại nhạc Việt và file MP3 đã được chia đoạn.

Các bản ghi từ hai nguồn được đưa vào pipeline lời bài hát. Vì quá trình gộp không khử trùng lặp tác phẩm xuyên nguồn, tổng 5.460 ở phần lời bài hát là **số bản ghi theo nguồn**, không khẳng định có 5.460 tác phẩm khác nhau.

## 2. Quy mô dữ liệu đã xử lý

### 2.1 Metadata từ Hugging Face

`CleanDataStream.py` tải các file Parquet trong thư mục `data/`, ghép dữ liệu và bỏ bản ghi trùng theo `id` (giữ bản ghi đầu tiên).

| Chỉ số | Kết quả |
|---|---:|
| Bản ghi sau khi ghép và khử trùng lặp theo `id` | 4.820 |
| Bản ghi đạt điều kiện hợp lệ | 4.820 |
| Bản ghi không hợp lệ | 0 |
| Có phụ đề (`subtitle_clean`) | 1.179 |
| Không có phụ đề | 3.641 |
| Thời lượng ghi nhận | 64–890 giây |

Điều kiện hợp lệ trong script: `id`, tiêu đề đã làm sạch và nghệ sĩ đã làm sạch phải có giá trị; thời lượng phải nằm trong khoảng 30 giây đến 15 phút. Ở lần chạy được lưu trong workspace, cả 4.820 bản ghi đều thỏa mãn điều kiện này.

Các trường metadata chính gồm ID, tiêu đề/nghệ sĩ gốc và đã làm sạch, thời lượng, lượt xem, lượt thích, ngày tải lên, phụ đề, trạng thái lời bài hát và nguồn lời bài hát. Đây là dữ liệu metadata trực tuyến; không có file audio tương ứng trong pipeline của script này.

### 2.2 Âm thanh Việt trong `Music/`

`cleanDataViet.py` quét đệ quy file `.mp3`, lấy thể loại từ thư mục cấp đầu tiên và gom các đoạn theo `track_id`.

| Thể loại | Track | Đoạn âm thanh |
|---|---:|---:|
| `cach_mang` | 108 | 993 |
| `nhac_trinh` | 138 | 1.203 |
| `rap_viet` | 158 | 1.053 |
| `rock_viet` | 129 | 1.123 |
| `tru_tinh` | 107 | 999 |
| **Tổng** | **640** | **5.371** |

Thông tin đọc được từ 5.371 file trong lần chạy hiện tại:

- Mỗi đoạn có thời lượng 30 giây, sample rate 22.050 Hz và 1 kênh.
- Kích thước mỗi file là 2.646.058 byte.
- Mỗi track có từ 2 đến 19 đoạn; trung bình khoảng 8,4 đoạn.
- 165 dòng ở cấp đoạn và 18 track được đánh dấu cần rà soát tên; nguyên nhân là tên title/artist không tách chắc chắn (159 đoạn/17 track) hoặc tên rỗng (6 đoạn/1 track).

Các giá trị audio được đọc từ header/metadata của file (WAV được phân tích RIFF; MP3 dùng Mutagen). Script không giải mã toàn bộ file để xác nhận chất lượng nghe hoặc tính toàn vẹn nội dung.

## 3. Làm sạch và các tệp đầu ra

### 3.1 `CleanDataStream.py`

- Chuẩn hóa văn bản Unicode về NFC và gom khoảng trắng.
- Làm sạch tiêu đề bằng cách bỏ một số cụm quảng bá như “official music video”, “official audio”, “lyrics”, “4K” và “HD”; bỏ ngoặc trống và dấu phân cách thừa.
- Làm sạch nghệ sĩ bằng cách bỏ “official”/“official channel” và khoảng trắng thừa.
- Chuẩn hóa giá trị phụ đề rỗng hoặc dạng chuỗi `none`, `null`, `nan` thành giá trị thiếu.
- Đánh dấu bản ghi không hợp lệ và lưu lý do nếu thiếu ID/tiêu đề/nghệ sĩ hoặc thời lượng ngoài ngưỡng.

Các đầu ra trong thư mục `data/`:

- `raw_metadata.csv` và `raw_metadata.parquet`: dữ liệu sau khi ghép và khử trùng lặp ID, có thêm các cột hỗ trợ kiểm tra.
- `clean_metadata.csv` và `clean_metadata.parquet`: các trường metadata được giữ lại cho bản ghi hợp lệ.
- `invalid_metadata.csv`: bản ghi không hợp lệ; file hiện có 0 dòng dữ liệu.

### 3.2 `cleanDataViet.py`

- Chuẩn hóa tên thư mục thể loại thành chữ thường và dấu gạch dưới.
- Bỏ hậu tố `_segment_N`, số thứ tự track và một số mã tải xuống/hậu tố chất lượng khỏi tên bài.
- Chuẩn hóa dấu gạch dưới, CamelCase và khoảng trắng; thử tách title/artist theo dấu ` - `.
- Ghi metadata cho từng đoạn, gồm đường dẫn, thể loại, `track_id`, số đoạn, kích thước, thời lượng, sample rate và số kênh.
- Gộp theo `track_id` để tạo dữ liệu cấp bài; thời lượng track là tổng thời lượng các đoạn được gom.
- Đánh dấu tên cần rà soát thay vì tự khẳng định các trường hợp phân tách mơ hồ.

Hai đầu ra do script tạo là `cleaned_segments.csv` và `cleaned_tracks_for_lyrics.csv`. Trong workspace hiện tại, các bản này nằm dưới `dataViet/`.

## 4. Lời bài hát và dữ liệu sẵn sàng cho vector hóa

Các báo cáo làm sạch lời bài hát hiện ghi nhận:

| Nguồn | Bản ghi track | Có lời tìm được | Được đề xuất vector hóa |
|---|---:|---:|---:|
| VMusic | 4.820 | 2.597 | 2.265 |
| `dataViet` | 640 | 344 | 335 |
| **Tổng theo nguồn** | **5.460** | **2.941** | **2.600** |

Ở tập VMusic, 332 bản ghi có lời cần rà soát hoặc không đạt điều kiện vector hóa bảo thủ; trong đó có các trường hợp lời rỗng sau làm sạch, nguồn chưa xác minh, điểm khớp thấp hoặc lời quá ngắn. Ở `dataViet`, 9 bản ghi có lời được gắn nhãn cần xem lại (5 điểm khớp thấp, 4 nguồn chưa xác minh). Các bản ghi không có lời không được đưa vào tập vector-ready.

Tệp `VMusic_lyrics_V3_2_cleaning_report.txt` ghi nhận 2.223 bản ghi chất lượng `missing` trong nguồn VMusic. Nhiều trạng thái truy xuất `api_error`/`provider_paused` được giữ nguyên; không nên hiểu lỗi dịch vụ là bằng chứng bài hát không có lời. Trong đầu ra kết hợp, 2.519 bản ghi được phân loại `missing` theo chất lượng lời.

Lời gốc được giữ riêng trong `lyrics`, nội dung đã làm sạch được lưu ở `lyrics_clean`. Các lời trùng được gắn cờ, không tự xóa vì bản cover/remix có thể dùng chung lời. Các tệp đáng chú ý:

- `data/VMusic_lyrics_V3_2_cleaned.csv` và `data/VMusic_lyrics_V3_2_vector_ready.csv`
- `dataViet/tracks_with_lyrics_cleaned.csv` và `dataViet/tracks_with_lyrics_vector_ready.csv`
- `data/lyrics_all_sources_cleaned.csv`: 5.460 dòng kết hợp; có thể còn bài trùng giữa hai nguồn.
- `data/VMusic_lyrics_V3_2_cleaning_report.json` và `.txt`: thống kê cho lần làm sạch VMusic.

## 5. Ý nghĩa đối với recommendation system

- **Content-based trên lời bài hát** là hướng có thể thử ngay với 2.600 bản ghi được đề xuất vector hóa. Cần lưu phiên bản mô hình/cấu hình vector và giữ lại thông tin nguồn, chất lượng lời để đánh giá kết quả.
- **Content-based trên âm thanh** hiện chỉ có thể áp dụng cho 640 track trong `dataViet` có các đoạn MP3 cục bộ. Cần trích xuất đặc trưng hoặc audio embedding nhất quán; không nên xem 4.820 dòng metadata VMusic là có audio sẵn.
- **Collaborative filtering/cá nhân hóa theo hành vi thực tế** chưa thể đánh giá vì chưa có lịch sử user thật. `Dataset/user data/synthetic_user_song_data.csv` có 120.000 tương tác mô phỏng (500 user) và chỉ phù hợp kiểm tra kỹ thuật; không đại diện cho hành vi người nghe.
- Ứng dụng hiện dùng `data/lyrics_all_sources_cleaned.csv` làm catalog kết hợp. File có 5.460 dòng nguồn; một dòng `dataViet` thiếu tiêu đề nên loader cảnh báo và bỏ qua, còn 5.459 track hợp lệ (4.820 VMusic, 639 `dataViet`). Có 2.600 track được duyệt lời để vector hóa.
- `Dataset/stats_music.csv` là corpus thể loại riêng (28.372 dòng, 7 thể loại), không có ID khớp catalog nên không dùng để huấn luyện/đánh giá recommender hiện tại.
- Khi chia train/validation/test, cần chia theo track/tác phẩm thay vì theo đoạn để các đoạn của cùng bài không rơi vào nhiều tập. Với dữ liệu kết hợp, cần rà soát trùng lặp giữa nguồn VMusic và `dataViet` trước khi đánh giá.

## 6. Giới hạn và việc cần làm tiếp

1. Rà soát thủ công các tên bài/artist đã được đánh dấu; phép tách theo dấu gạch ngang là heuristic và có thể gán sai.
2. Xác minh trùng lặp xuyên nguồn bằng metadata đáng tin cậy; ID của hai pipeline khác định dạng nên không tự bảo đảm nhận diện cùng một tác phẩm.
3. Kiểm tra khả năng giải mã đầy đủ và nghe mẫu các file âm thanh, đặc biệt vì tất cả đoạn hiện có cùng thời lượng và kích thước.
4. Kiểm tra các nguồn lời, match score và bản ghi được đánh dấu review trước khi dùng cho huấn luyện; phân biệt `api_error` với `missing`.
5. Thu thập dữ liệu tương tác người dùng nếu mục tiêu là cá nhân hóa hoặc collaborative filtering.
6. `cleanDataViet.py` hiện đặt `DATASET_ROOT` bằng đường dẫn tuyệt đối `D:\0_College\#DS300\Music`; cần sửa cấu hình này nếu chạy ở máy hoặc thư mục khác.

## 7. Chạy lại các bước tạo metadata

Chạy từ thư mục gốc dự án:

```powershell
python .\CleanDataStream.py
python .\cleanDataViet.py
```

`CleanDataStream.py` cần kết nối Hugging Face và ghi các đầu ra vào `data/`. `cleanDataViet.py` dùng đường dẫn `Music/` được khai báo cố định trong script và ghi hai CSV đầu ra vào thư mục làm việc hiện tại. Các bước lấy/làm sạch lời bài hát là pipeline riêng; báo cáo và số liệu ở trên phản ánh các tệp đầu ra hiện có trong workspace, không phải kết quả của việc chạy lại toàn bộ pipeline trong lúc viết báo cáo.
