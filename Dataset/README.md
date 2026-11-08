Cách chạy file lấy lyrics:

- `getLyrics.py` đọc `data/clean_metadata_no_timestamp.csv` và ghi kết quả vào `data/clean_metadata_with_lyrics.csv` cùng file Parquet. Thư mục `data/` được script tự tạo; cần tạo/copy file input vào đúng đường dẫn trước khi chạy.
- `getLyricsViet.py` đọc `dataViet/cleaned_tracks_for_lyrics.csv` và ghi kết quả vào `dataViet/tracks_with_lyrics.csv`.
- Cả hai script lưu checkpoint và tiếp tục từ file output nếu file đó đã tồn tại; không cần đổi tên output thành input để chạy tiếp. Đảm bảo input và output hiện có thuộc cùng một bộ dữ liệu.

===================================
Các file clean / tiền xử lý

1. `CleanDataStream.py`: tải metadata từ Hugging Face, chuẩn hóa title/artist/subtitle và tạo `data/raw_metadata.csv`, `data/clean_metadata.csv`, `data/invalid_metadata.csv` cùng các bản Parquet. Script cần kết nối Internet; điều kiện lọc và đường dẫn đầu ra được khai báo trong code.

2. `recleanTimestamp.py`: đọc `data/clean_metadata.csv`, loại timestamp và metadata WEBVTT khỏi `subtitle_clean`, sau đó ghi `data/clean_metadata_no_timestamp.csv`.

3. `cleanDataViet.py`: quét các file MP3 trong `Music/`, chuẩn hóa tên file, đọc metadata audio, đánh dấu tên cần rà soát và tạo CSV theo đoạn (`cleaned_segments.csv`) và theo bài (`cleaned_tracks_for_lyrics.csv`). `DATASET_ROOT` hiện được khai báo bằng đường dẫn tuyệt đối trong script; hai file CSV được ghi vào thư mục làm việc hiện tại.

4. `clean_lyrics.py`: làm sạch lời bài hát nhưng giữ nguyên cột `lyrics` gốc; thêm `lyrics_clean`, nhãn chất lượng, cờ rà soát và cờ `lyrics_use_for_vector`. Mặc định đọc `data/VMusic_lyrics_checkpoint_V3_2.csv`, đồng thời xử lý `dataViet/tracks_with_lyrics.csv`, tạo dữ liệu kết hợp, các file review/vector-ready và báo cáo JSON/TXT.

   ```powershell
   python .\clean_lyrics.py
   ```

   Có thể chỉ định file đầu vào hoặc thư mục đầu ra:

   ```powershell
   python .\clean_lyrics.py --input .\data\VMusic_lyrics_checkpoint_V3_2.csv --output-dir .\data
   ```

## Thứ tự chạy tham khảo

1. Chạy `CleanDataStream.py` để tạo metadata VMusic.
2. Chạy `recleanTimestamp.py` để chuẩn hóa phụ đề trước khi tìm lời.
3. Chạy `cleanDataViet.py` từ thư mục `dataViet` để các CSV đầu ra nằm đúng vị trí `getLyricsViet.py` mong đợi:

   ```powershell
   Push-Location .\dataViet
   python ..\cleanDataViet.py
   Pop-Location
   ```

4. Chạy `getLyrics.py` với dữ liệu trong `data/` và `getLyricsViet.py` với dữ liệu trong `dataViet/` để lấy lời bài hát.
5. Chạy `clean_lyrics.py` để làm sạch lời, tạo tập review và tập vector-ready.

Các script lấy lời cần kết nối dịch vụ LRCLIB. Kết quả phụ thuộc vào trạng thái dịch vụ và mức độ khớp title/artist; hãy kiểm tra file review trước khi dùng dữ liệu để huấn luyện. Tên file input/output và cách tiếp tục từ checkpoint có thể khác theo từng script, vì vậy hãy kiểm tra phần cấu hình đầu file trước khi chạy.

## Danh sách file CSV

### Thư mục `data/`

| File | Nội dung |
|---|---|
| `raw_metadata.csv` | Metadata VMusic sau khi ghép các file nguồn và khử trùng lặp theo `id`; gồm dữ liệu gốc và các cột kiểm tra. |
| `clean_metadata.csv` | Metadata hợp lệ đã làm sạch title/artist; đầu vào cho bước loại timestamp. |
| `invalid_metadata.csv` | Các metadata không đạt điều kiện hợp lệ, kèm lý do. |
| `clean_metadata_no_timestamp.csv` | Metadata sau khi làm sạch timestamp trong phụ đề; đầu vào mặc định của `getLyrics.py`. |
| `clean_metadata_with_lyrics.csv` | Metadata VMusic đã bổ sung lời bài hát và trạng thái/nguồn tìm lời. |
| `VMusic_lyrics_checkpoint_V3_2.csv` | Bản checkpoint lời VMusic dùng làm đầu vào cho bước làm sạch lời V3.2. |
| `VMusic_lyrics_V3_2_cleaned.csv` | Toàn bộ bản ghi VMusic sau làm sạch lời, có thêm nhãn chất lượng và cờ rà soát/vector hóa. |
| `VMusic_lyrics_V3_2_lyrics_review.csv` | Các bản ghi VMusic có lời cần kiểm tra thủ công. |
| `VMusic_lyrics_V3_2_vector_ready.csv` | Tập VMusic được đề xuất sử dụng để vector hóa lời. |
| `lyrics_all_sources_cleaned.csv` | Lời đã làm sạch, gộp các bản ghi từ VMusic và `dataViet`. |
| `lyrics_all_sources_with_missing_filled.csv` | Bản kết hợp sau bước thử bổ sung lời cho các bản ghi còn thiếu. |
| `lyrics_all_sources_with_missing_filled_checkpoint.csv` | Checkpoint của bước bổ sung lời cho dataset kết hợp. |

### Thư mục `dataViet/`

| File | Nội dung |
|---|---|
| `cleaned_segments.csv` | Metadata theo từng đoạn MP3, gồm đường dẫn, track, số đoạn và thông tin audio. |
| `cleaned_tracks_for_lyrics.csv` | Metadata đã gom theo track; đầu vào của `getLyricsViet.py`. |
| `tracks_with_lyrics.csv` | Metadata track Việt sau khi tìm lời; đồng thời là checkpoint để tiếp tục chạy. |
| `tracks_with_lyrics_cleaned.csv` | Track Việt sau làm sạch lời và gắn nhãn chất lượng. |
| `tracks_with_lyrics_lyrics_review.csv` | Các track Việt có lời cần kiểm tra thủ công. |
| `tracks_with_lyrics_vector_ready.csv` | Các track Việt được đề xuất sử dụng để vector hóa lời. |

## Tổng kết: các file final cần dùng

- **Vector hóa lời VMusic:** `data/VMusic_lyrics_V3_2_vector_ready.csv`.
- **Vector hóa lời từ nhạc Việt trong `Music/`:** `dataViet/tracks_with_lyrics_vector_ready.csv`.
- **Dùng chung hai nguồn lời:** có thể ghép hai file vector-ready trên; giữ cột `dataset_source` để truy dấu nguồn và kiểm tra trùng lặp trước khi đánh giá mô hình. Không dùng toàn bộ `lyrics_all_sources_cleaned.csv` thay cho tập vector-ready nếu chưa lọc `lyrics_use_for_vector == True`.
- **Thử nghiệm trên âm thanh:** dùng các MP3 trong `Music/` cùng metadata đoạn ở `dataViet/cleaned_segments.csv`; metadata gộp theo track nằm ở `dataViet/cleaned_tracks_for_lyrics.csv`. Các file này phục vụ pipeline audio, không phải corpus lời đã vector hóa.
- **Metadata VMusic:** nếu cần thông tin title/artist, lượt xem, lượt thích hoặc thời lượng, dùng `data/clean_metadata.csv` (hoặc `data/clean_metadata_with_lyrics.csv` nếu cần trạng thái/lời đã tìm). Đây là metadata, không có audio cục bộ.

Các file `*_checkpoint*.csv`, `*_lyrics_review.csv` và `*_cleaned.csv` là file tiến trình, rà soát hoặc trung gian; chỉ dùng khi cần tiếp tục xử lý, kiểm tra chất lượng hoặc tái tạo tập final.

## Các file recommendation system đang sử dụng

- Catalog huấn luyện/runtime: `data/lyrics_all_sources_cleaned.csv`. Giữ tất cả track hợp lệ từ VMusic và `dataViet`; lời chỉ được dùng khi `lyrics_use_for_vector` là true. File hiện có 5.460 dòng nguồn, trong đó một dòng `dataViet` thiếu tiêu đề; loader cảnh báo và bỏ dòng đó, nên catalog runtime có 5.459 track.
- Hai file `data/VMusic_lyrics_V3_2_vector_ready.csv` và `dataViet/tracks_with_lyrics_vector_ready.csv` là đầu ra đã duyệt để tái tạo corpus lời. Các model runtime đọc file kết hợp ở trên.
- `user data/synthetic_user_song_data.csv` có 500 tài khoản mô phỏng; chỉ dùng kiểm tra loader/demo, không dùng để khẳng định độ chính xác hoặc hành vi người nghe thật.
- `stats_music.csv` là dataset thể loại độc lập, không có ID chung với catalog VMusic/dataViet; không đưa vào recommender hiện tại.
- `lyrics_all_sources_with_missing_filled.csv` không phải catalog mặc định và không nên dùng thay catalog cleaned nếu chưa kiểm tra trạng thái/nguồn lời.

Nếu chỉ giữ những file cần cho model đã train, có thể giữ catalog kết hợp, mã trong thư mục dự án và `trained_artifacts/`. Giữ các file vector-ready, script xử lý, metadata gốc và báo cáo nếu cần tái tạo hoặc kiểm toán dữ liệu. Có thể bỏ `stats_music.csv` và các bản trung gian/checkpoint nếu không dùng cho nghiên cứu khác; không xóa dữ liệu nguồn trước khi sao lưu.
