import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import glob

# 🔹 Đường dẫn tới thư mục log TensorBoard bạn đã tải về
log_dir = r"\matcha_vi\tensorboard\mel80band"

# 🔹 Tìm file event mới nhất
event_files = sorted(glob.glob(f"{log_dir}\\events.out.tfevents.*"), key=lambda x: -os.path.getmtime(x))
event_file = event_files[0]
print("Using event file:", event_file)

# 🔹 Load log
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 🔹 Chọn tag bạn muốn vẽ
tags = [
    'sub_loss/train_dur_loss_step',
    'sub_loss/train_prior_loss_step',
    'sub_loss/train_diff_loss_step',
    'loss/train_step'
]

# 🔹 Vẽ tất cả trên cùng 1 biểu đồ
plt.figure(figsize=(10, 6))

for tag in tags:
    if tag in ea.Tags()['scalars']:
        data = ea.Scalars(tag)
        steps = [x.step for x in data]
        values = [x.value for x in data]
        plt.plot(steps, values, label=tag)
    else:
        print(f"⚠️ Không tìm thấy tag: {tag}")

plt.title("Training Loss Curves")
plt.xlabel("Step")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()
