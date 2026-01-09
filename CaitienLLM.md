# Cải tiến LLM cho IPIATTS (mô tả kỹ thuật)

Tài liệu này tập trung vào bản chất hoạt động kỹ thuật của LLM (PhoBERT) trong pipeline Matcha-TTS của IPIATTS: tín hiệu đi đâu, biến nào được tính, và vì sao mô-đun prosody nâng chất lượng.

## Kiến trúc và dòng tín hiệu
- **Đầu vào song song**: phoneme ids (từ filelist) và `raw_texts` (câu gốc) đi vào hai nhánh: encoder truyền thống và LLM PhoBERT.
- **Encoder chuẩn (TextEncoder)**: nhúng phoneme → stack conv/attention → xuất `mu_x` (đặc trưng text) và `logw` (log-duration từng token) cùng mask `x_mask`.
- **Nhánh LLM prosody**: PhoBERT mã hóa câu, lấy token CLS, chiếu tuyến tính sang không gian `prosody_dim`, rồi broadcast theo chiều dài phoneme để tạo tensor prosody `[B, prosody_dim, T_text]`.
- **Prosody Fusion**: cross-attention + gating: text làm query, prosody làm key/value; sau đó gate điều chỉnh mức ảnh hưởng và conv 1x1 hợp nhất thành `fused_text` cùng kích thước với encoder output.
- **Duration/Alignment**: dùng `logw` → `w = exp(logw) * x_mask` → làm tròn và nhân `length_scale` để quyết định độ dài mel, sinh path align (MAS nếu không có durations).
- **Decoder CFM**: nhận đặc trưng đã hợp nhất (text+prosody) và mask mel, thực hiện conditional flow matching để sinh mel-spectrogram; HifiGAN suy ra waveform.

## Công thức và tensor chính
- **LLM embedding**: PhoBERT cho `h_cls ∈ R^{B×H}`; chiếu: `p = W_p h_cls`, `p ∈ R^{B×D}` với `D = prosody_dim`. Broadcast: `P = repeat(p, T_text)` → `P ∈ R^{B×D×T}`.
- **Cross-attention trong ProsodyFusion**:
	- Query `Q = W_q F_text`, Key `K = W_k P`, Value `V = W_v P` (mọi thứ shape `B×C×T`).
	- Điểm chú ý: $A = \text{softmax}(Q^T K / \sqrt{C}) \in \mathbb{R}^{B\times T\times T}$ (mask chặn padding bằng -1e4 để ổn định FP16).
	- Prosody attend: `P_att = (A V^T)^T` → `B×C×T`.
	- Gate: `g = σ(W_g[ F_text ; P_att ])`; hợp nhất: `F_fused = FusionNet([F_text ; g ⊙ P_att])`.
- **Duration và align**: `w = exp(logw)`, `y_len = Σ ceil(w)`, `attn = MAS(mu_x, y, mask)`; loss thời lượng: $L_{dur} = \text{MSE}(\log w, \log w_{MAS})$.
- **Flow matching loss**: decoder học ánh xạ ODE; loss khuếch tán/CFM áp dụng trên segment mel đã mask.
- **Prior loss**: $L_{prior} = \frac{1}{N} ||y - \mu_y||_2^2$ giúp encoder gần mel thực.
- **Tổng loss**: $L = L_{dur} + L_{prior} + L_{cfm}$ (trọng số hiện để 1, có thể chỉnh nếu cần).

## Hành vi trong huấn luyện
- **Đóng băng PhoBERT mặc định**: chỉ học `W_p` và các lớp fusion, giảm VRAM và tránh huỷ ngữ nghĩa gốc của PhoBERT.
- **Truncation an toàn**: giới hạn `max_position_embeddings` của PhoBERT; nếu input vượt, cắt và log cảnh báo, tránh lỗi pos-ids khi FP16.
- **Dữ liệu bắt buộc**: batch phải cung cấp `raw_texts`; nếu không, prosody rơi về zero-tensor, mô hình vẫn chạy nhưng mất hiệu ứng LLM.
- **Multi-speaker**: khi `n_spks>1`, encoder nhận thêm `spk_emb`, decoder CFM cũng được điều kiện hoá speaker song song với prosody.

## Hành vi trong suy luận
- `MatchaTTS.synthesise` gọi lại nhánh PhoBERT (nếu có `raw_texts`) và cùng luồng align như training. `length_scale` tác động trực tiếp vào `w_ceil` để kéo giãn/thu tốc độ nói. `temperature` và `n_timesteps` điều khiển độ ngẫu nhiên và độ mượt của quỹ đạo flow.
- Nếu thiếu `raw_texts`, prosody = 0 → hành vi giống baseline không LLM.

## Độ phức tạp và footprint
- **Chi phí thêm**: một forward PhoBERT (BERT-base ~12 layer) + conv 1x1 và attention nhỏ. VRAM tăng chủ yếu ở hidden states PhoBERT; bật `freeze_llm` giảm gradient memory.
- **Tỷ lệ thời gian**: với batch nhỏ (1-4) và seq < 128 token, PhoBERT thường chiếm phần lớn latency CPU/GPU; decoder CFM còn lại ~O(T_mel × C^2) nhưng đã tối ưu cho inference.

## Các nút chỉnh quan trọng
- `llm_model_name`: đổi PhoBERT/XLM-R, phải tương thích tokenizer BPE.
- `prosody_dim`, `fusion_channels`: tăng để giữ nhiều thông tin nhịp/nhấn; trade-off VRAM.
- `freeze_llm`: đặt `False` để fine-tune; nên bật gradient checkpointing nếu GPU nhỏ.
- `use_attention` trong ProsodyFusion: có thể tắt để biến thành cộng đơn giản khi cần tốc độ.

## Kịch bản thất bại và fallback
- Không tải được PhoBERT hoặc thiếu `raw_texts`: dùng zero-prosody (mềm dẻo, không crash).
- Chuỗi quá dài: bị truncate; nếu chất lượng giảm, hãy cắt câu hoặc dùng model có `max_position_embeddings` lớn hơn.
- OOM PhoBERT: giảm batch, bật FP16, hoặc chuyển sang `SimpleProsodyAnalyzer` (không LLM).

## Tóm tắt lợi ích kỹ thuật
- Thêm tín hiệu ngữ nghĩa/nhấn nhá toàn câu vào encoder trước bước align, giúp MAS và duration ổn định hơn.
- Cross-attention + gating cho phép điều chỉnh mức ảnh hưởng, tránh việc prosody lấn át nội dung.
- Giữ tính tương thích ngược: pipeline vẫn chạy khi không có LLM, chỉ mất lợi ích prosody.
