# 🎯 Hướng Dẫn Training với train_matcha_kaggle.ipynb

## ✅ **ĐÃ CẬP NHẬT - Các Cải Tiến Đã Được Tích Hợp**

Khi bạn chạy `train_matcha_kaggle.ipynb`, **TẤT CẢ 3 CẢI TIẾN** sẽ được áp dụng tự động:

### **1️⃣ Token-Level Prosody Alignment** ✅
- Mỗi phoneme có prosody vector riêng
- **Mặc định:** `use_token_level_prosody=True`

### **2️⃣ PhoBERT Fine-tuning Support** ✅  
- Hỗ trợ fine-tune PhoBERT
- **Mặc định:** `finetune_llm=False` (để tiết kiệm GPU trên Kaggle)

### **3️⃣ Pause & Boundary Detection** ✅
- Tự động bật pause predictor
- Tự động bật boundary detector

---

## 📊 **So Sánh Config**

### **train_matcha_prosody.py** (Standalone script):
```python
CONFIG = {
    "use_token_level_prosody": True,
    "finetune_llm": False,  # Có thể bật True
}
```

### **train_matcha_kaggle.ipynb** (Notebook cho Kaggle):
```python
CONFIG = {
    "use_token_level_prosody": True,  # ✅ ĐÃ CÓ
    "finetune_llm": False,             # ✅ ĐÃ CÓ
}
```

**→ HOÀN TOÀN GIỐNG NHAU!**

---

## 🚀 **Cách Chạy Training**

### **Trên Kaggle (2 x T4 16GB):**

1. Upload notebook `train_matcha_kaggle.ipynb`
2. Chạy tất cả cells theo thứ tự
3. Model sẽ được train với config:
   ```
   ✅ Token-level prosody: TRUE
   ✅ Fine-tune PhoBERT: FALSE (tiết kiệm memory)
   ✅ Pause predictor: AUTO ENABLED
   ✅ Boundary detector: AUTO ENABLED
   ✅ Batch size: 16 (per 2 GPUs = 32 effective)
   ✅ DDP strategy: 2 devices
   ```

### **Trên Local Machine:**

#### **GPU Yếu (< 16GB):**
```bash
# Không cần sửa gì, config mặc định đã tối ưu
python train_matcha_final.py
```

#### **GPU Mạnh (>= 24GB):**
```python
# Sửa trong notebook cell tạo script:
CONFIG = {
    "finetune_llm": True,  # BẬT FINE-TUNING
    "batch_size": 8,        # Giảm batch size
}
```

---

## ⚙️ **Config Chi Tiết**

### **Cell tạo script trong notebook:**

```python
CONFIG = {
    # Prosody settings
    "llm_model_name": "vinai/phobert-base",
    "prosody_dim": 256,
    "use_token_level_prosody": True,  # ← CẢI TIẾN 1
    "finetune_llm": False,             # ← CẢI TIẾN 2 (tắt cho Kaggle)
    
    # Pause & Boundary tự động có trong model
    # Không cần config thêm
}
```

### **Khởi tạo model:**

```python
model = MatchaTTS(
    # ... các params khác ...
    llm_model_name=CONFIG["llm_model_name"],
    prosody_dim=CONFIG["prosody_dim"],
    use_token_level_prosody=CONFIG.get("use_token_level_prosody", True),
    finetune_llm=CONFIG.get("finetune_llm", False),
)
```

---

## 📈 **Losses Sẽ Thấy Trong Training**

Khi train với notebook, bạn sẽ thấy các losses sau:

```
Epoch 1/20:
├─ dur_loss: 2.45        # Duration prediction
├─ prior_loss: 1.23      # Encoder-decoder alignment
├─ diff_loss: 0.89       # Flow matching diffusion
├─ acoustic_loss: 0.45   # TỔNG CỦA:
│  ├─ pitch_loss: 0.12   #   Pitch prediction
│  ├─ energy_loss: 0.11  #   Energy prediction
│  ├─ pause_loss: 0.08   #   ✅ Pause prediction (MỚI)
│  └─ boundary_loss: 0.05#   ✅ Boundary detection (MỚI)
└─ total_loss: 5.02
```

---

## 🔍 **Kiểm Tra Cải Tiến Có Hoạt Động Không**

### **Trong TensorBoard:**

```bash
tensorboard --logdir outputs/logs
```

Bạn sẽ thấy:
- `train/pause_loss` ← Nếu thấy loss này → Pause predictor hoạt động ✅
- `train/boundary_loss` ← Nếu thấy loss này → Boundary detector hoạt động ✅

### **Trong Console Log:**

```
PhoBERT fine-tuning ENABLED/FROZEN
# Nếu thấy FROZEN → finetune_llm=False ✅
# Nếu thấy ENABLED → finetune_llm=True ✅
```

---

## 💡 **Khuyến Nghị Cấu Hình**

### **Kaggle (2 x T4 16GB):**
```python
{
    "use_token_level_prosody": True,   # ✅ BẬT
    "finetune_llm": False,              # ❌ TẮT (T4 không đủ mạnh)
    "batch_size": 16,                   # Tối ưu cho 2 GPU
    "devices": 2,
    "strategy": "ddp",
}
```

### **Colab Pro+ (A100 40GB):**
```python
{
    "use_token_level_prosody": True,   # ✅ BẬT
    "finetune_llm": True,               # ✅ BẬT (A100 đủ mạnh)
    "batch_size": 32,
    "devices": 1,
}
```

### **Local RTX 4090 (24GB):**
```python
{
    "use_token_level_prosody": True,   # ✅ BẬT
    "finetune_llm": True,               # ✅ BẬT
    "batch_size": 16,
    "devices": 1,
}
```

---

## ⚠️ **Troubleshooting**

### **Out of Memory khi train:**

**Giải pháp 1:** Giảm batch size
```python
CONFIG["batch_size"] = 8  # Giảm từ 16 xuống 8
```

**Giải pháp 2:** Tắt fine-tuning
```python
CONFIG["finetune_llm"] = False
```

**Giải pháp 3:** Dùng gradient accumulation
```python
# Trong Trainer:
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Effective batch = 8 * 4 = 32
)
```

### **Loss không giảm:**

1. Kiểm tra data statistics có đúng không
2. Kiểm tra pitch/energy có được normalize không
3. Thử learning rate thấp hơn: `1e-5`

---

## 📋 **Checklist Trước Khi Train**

- [ ] Data đã được chuẩn bị (3 columns: audio|text_vi|text_ipa)
- [ ] Mel statistics đã được tính (CALCULATED_MEAN, CALCULATED_STD)
- [ ] Pitch/Energy statistics đã được tính
- [ ] Audio files tồn tại tại đường dẫn chỉ định
- [ ] PhoBERT model có thể download được (internet connection)
- [ ] GPU memory >= 16GB (hoặc dùng batch_size nhỏ hơn)

---

## ✅ **KẾT LUẬN**

**CÂU TRẢ LỜI:** Có! Khi bạn dùng `train_matcha_kaggle.ipynb` để train, **TẤT CẢ các cải tiến** từ `train_matcha_prosody.py` đều được áp dụng:

1. ✅ **Token-level prosody** - Đã tích hợp
2. ✅ **PhoBERT fine-tuning** - Có hỗ trợ (mặc định tắt cho Kaggle)
3. ✅ **Pause predictor** - Tự động có trong model
4. ✅ **Boundary detector** - Tự động có trong model

**Không có gì khác biệt!** Cả 2 file đều dùng chung model architecture và config.

---

## 🎓 **Tài Liệu Tham Khảo**

- [PROSODY_IMPROVEMENTS.md](PROSODY_IMPROVEMENTS.md) - Chi tiết về 3 cải tiến
- [train_matcha_prosody.py](train_matcha_prosody.py) - Standalone training script
- [train_matcha_kaggle.ipynb](train_matcha_kaggle.ipynb) - Notebook cho Kaggle/Colab

---

**Happy Training! 🎤**
