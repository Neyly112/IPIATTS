import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
FOLDER_REF = "ket_qua_test_full_pack_1/ref"
FOLDER_GEN = "ket_qua_test_full_pack_1/gen"
OUTPUT_FOLDER = "mel_images"   # thư mục lưu ảnh mel

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def plot_mel(real_path, gen_path, out_path):
    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    fmax = 8000

    try:
        y_real, _ = librosa.load(real_path, sr=sr)
        y_gen, _ = librosa.load(gen_path, sr=sr)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {real_path} hoặc {gen_path}")
        return

    def get_spec(y):
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax
        )
        return librosa.power_to_db(S, ref=np.max)

    S_real = get_spec(y_real)
    S_gen = get_spec(y_gen)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

    img1 = librosa.display.specshow(S_real, x_axis='time', y_axis='mel',
                                    sr=sr, hop_length=hop_length, fmax=fmax,
                                    ax=ax[0], cmap='viridis')
    ax[0].set(title='Ground Truth (Giọng thật)')
    ax[0].label_outer()

    img2 = librosa.display.specshow(S_gen, x_axis='time', y_axis='mel',
                                    sr=sr, hop_length=hop_length, fmax=fmax,
                                    ax=ax[1], cmap='viridis')
    ax[1].set(title='Synthesized (Matcha-TTS)')
    ax[1].set(xlabel='Time (s)')

    fig.colorbar(img1, ax=ax, format='%+2.0f dB', label='Decibels')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)  # đóng figure để tránh chiếm bộ nhớ

    print(f"Đã lưu ảnh: {out_path}")

if __name__ == "__main__":
    # Lấy danh sách file ref
    ref_files = glob.glob(os.path.join(FOLDER_REF, "*.wav"))

    for ref_file in ref_files:
        fname = os.path.basename(ref_file)
        gen_file = os.path.join(FOLDER_GEN, fname)

        if not os.path.exists(gen_file):
            print(f"Không tìm thấy file gen cho {fname}")
            continue

        out_image = os.path.join(OUTPUT_FOLDER, fname.replace(".wav", "_mel.png"))
        plot_mel(ref_file, gen_file, out_image)

    print("=== HOÀN THÀNH VẼ MEL-SPECTROGRAM CHO NHIỀU FILE ===")
