import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN (CHỈ CẦN ĐỔI CHỖ NÀY)
# ============================================================
log_dir = r"D:\Bai Tap\DACNTT\logs_new_voice600\matcha_vi\tensorboard\mel80band"

# ============================================================
# 1. Tìm và sắp xếp file logs
# ============================================================
event_files = sorted(
    glob.glob(os.path.join(log_dir, "events.out.tfevents.*")),
    key=lambda x: os.path.getmtime(x)
)

print(f"Found {len(event_files)} event files:")
for f in event_files:
    print(" -", os.path.basename(f)) # Chỉ in tên file cho gọn

# ============================================================
# 2. Helper: Gộp dữ liệu từ nhiều file
# ============================================================
def load_and_merge_tags(event_files, tag):
    merged_steps = []
    merged_values = []

    for file in event_files:
        # size_guidance để đảm bảo load hết scalars, không bị cắt bớt
        ea = event_accumulator.EventAccumulator(file, size_guidance={event_accumulator.SCALARS: 0})
        try:
            ea.Reload()
        except Exception as e:
            print(f"⚠️ Lỗi load file {file}: {e}")
            continue

        available_tags = ea.Tags().get('scalars', [])
        
        if tag in available_tags:
            data = ea.Scalars(tag)
            merged_steps.extend([x.step for x in data])
            merged_values.extend([x.value for x in data])

    if not merged_steps:
        print(f"⚠️ Tag không tồn tại: {tag}")
        return None, None

    # Sắp xếp theo step để đảm bảo thứ tự đúng
    combined = list(zip(merged_steps, merged_values))
    combined.sort(key=lambda x: x[0])
    
    # Unzip về lại 2 list
    merged_steps, merged_values = zip(*combined)
    return list(merged_steps), list(merged_values)

# ============================================================
# 3. Helper: Vẽ biểu đồ (ĐÃ SỬA LOGIC EPOCH)
# ============================================================
def plot_tag(tag, title, ylabel):
    steps, values = load_and_merge_tags(event_files, tag)

    if steps is None: return

    plt.figure(figsize=(10, 5))

    # --- LOGIC MỚI: Tự động chuyển sang Epoch ---
    if "epoch" in tag.lower():
        # Nếu là tag epoch, ta giả định mỗi điểm dữ liệu là 1 epoch
        # Tạo list [1, 2, 3, ..., N]
        x_axis = range(1, len(values) + 1)
        xlabel = "Epoch"
        print(f"✅ Vẽ {tag}: Phát hiện {len(values)} epochs.")
    else:
        # Nếu là step hoặc grad norm, giữ nguyên step
        x_axis = steps
        xlabel = "Global Step"
    # ---------------------------------------------

    plt.plot(x_axis, values, linewidth=1.5)
    
    # Trang trí biểu đồ
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu ảnh tự động (nếu muốn, bỏ comment dòng dưới)
    # plt.savefig(f"{title}.png") 
    
    plt.show()

# ============================================================
# 4. VẼ BIỂU ĐỒ (CHỈ VẼ CÁC TAG _EPOCH CHO ĐẸP)
# ============================================================

print("\n--- Đang vẽ biểu đồ ---")

# 1) TOTAL LOSS
# Chỉ vẽ _epoch để trục ngang là Epoch 1, 2, 3...
plot_tag("loss/train_epoch", "Training Total Loss (per epoch)", "Loss")

# 2) DURATION LOSS
plot_tag("sub_loss/train_dur_loss_epoch", "Duration Loss (per epoch)", "Duration Loss")

# 3) DIFFUSION / RECON LOSS
plot_tag("sub_loss/train_diff_loss_epoch", "Diffusion / Recon Loss (per epoch)", "Diff Loss")

# 4) PRIOR LOSS
plot_tag("sub_loss/train_prior_loss_epoch", "Prior Loss (per epoch)", "Prior Loss")

# 5) GRAD NORM (Cái này thường lưu theo Step, vẫn giữ nguyên)
plot_tag("grad_norm/grad_2.0_norm_total", "Gradient Norm", "Grad Norm")

# --- Lưu ý ---
# Các dòng plot_tag("..._step") đã được bỏ đi vì:
# 1. Nó quá nhiễu (zig-zag).
# 2. Rất khó chuyển Step -> Epoch chính xác nếu không biết Batch Size.
# Dùng các tag "..._epoch" ở trên là chuẩn nhất cho báo cáo.