import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
import json
from tqdm import tqdm

def calculate_rms(signal):
    """计算信号的有效值(RMS)"""
    return np.sqrt(np.mean(signal**2))

def main():
    # ================= 配置区 =================
    clean_dir = r"/root/SEMamba-main/data/树蛙声音"      # 存放33个树蛙声音的文件夹
    noise_dir = r"/root/SEMamba-main/data/100个机场"
    output_dir = "data/processed"    # 处理后的输出路径
    
    snr_list = [-15, -10, -5, 0, 5]    # 5种信噪比 (dB)
    duration = 10                    # 固定片段时长 (秒)
    chunks_per_noise = 6             # 每个机场噪声切出 6 段
    calls_per_chunk = 10             # 每 10 秒内放置 10 个蛙鸣 (策略A)
    sr = 16000                       # 采样率
    # ==========================================
    
    out_clean_dir = os.path.join(output_dir, "clean")
    out_noisy_dir = os.path.join(output_dir, "noisy")
    os.makedirs(out_clean_dir, exist_ok=True)
    os.makedirs(out_noisy_dir, exist_ok=True)
    
# 强制获取绝对路径，防止终端执行位置不对
    clean_dir_abs = os.path.abspath(clean_dir)
    noise_dir_abs = os.path.abspath(noise_dir)
    
    print("\n" + "="*40)
    print("【系统路径诊断信息】")
    print(f"当前 Python 正在这个目录下运行: {os.getcwd()}")
    print(f"树蛙文件夹目标路径: {clean_dir_abs}  (存在状态: {os.path.exists(clean_dir_abs)})")
    print(f"机场文件夹目标路径: {noise_dir_abs}  (存在状态: {os.path.exists(noise_dir_abs)})")
    print("="*40 + "\n")

    clean_files = []
    noise_files = []
    
    # 兼容处理：无论后缀是 .wav 还是 .WAV 都能找到
    if os.path.exists(clean_dir_abs):
        clean_files = [os.path.join(clean_dir_abs, f) for f in os.listdir(clean_dir_abs) if f.lower().endswith('.wav')]
    if os.path.exists(noise_dir_abs):
        noise_files = [os.path.join(noise_dir_abs, f) for f in os.listdir(noise_dir_abs) if f.lower().endswith('.wav')]

    if not clean_files or not noise_files:
        print(f"❌ 致命错误：")
        print(f"树蛙文件夹中找到 {len(clean_files)} 个 wav 文件。")
        print(f"机场文件夹中找到 {len(noise_files)} 个 wav 文件。")
        print("请检查文件夹是否存在，并且里面确实装有 .wav 格式的音频文件！")
        return
        
    print(f"找到 {len(clean_files)} 个树蛙音频，{len(noise_files)} 个机场噪声。")
    print("开始加载并预处理音频数据...")

    # 1. 预先加载所有的纯净树蛙叫声
    frog_wavs = []
    for c_file in clean_files:
        wav, _ = librosa.load(c_file, sr=sr)
        frog_wavs.append(wav)
        
    # 2. 顺序加载机场噪声，并严格按 10s 切割出 600 个噪声片段
    target_length = duration * sr
    noise_chunks = []
    
    for n_file in noise_files:
        n_name = os.path.splitext(os.path.basename(n_file))[0]
        n_wav, _ = librosa.load(n_file, sr=sr)
        
        # 尝试提取前 60 秒的数据，切成 6 段
        for i in range(chunks_per_noise):
            start_idx = i * target_length
            end_idx = start_idx + target_length
            
            # 提取切片
            if end_idx <= len(n_wav):
                chunk = n_wav[start_idx:end_idx]
            else:
                # 如果单个文件不够长，用全零补齐到 10 秒
                chunk = np.zeros(target_length)
                available = len(n_wav) - start_idx
                if available > 0:
                    chunk[:available] = n_wav[start_idx:]
            
            noise_chunks.append((f"{n_name}_part{i+1}", chunk))

    print(f"成功切分出 {len(noise_chunks)} 个不重复的背景噪声片段。")
    
    # 3. 开始合成
    clean_json_dict = {}
    noisy_json_dict = {}
    total_files = len(snr_list) * len(noise_chunks)
    file_idx = 0
    
    with tqdm(total=total_files, desc="合成数据中") as pbar:
        for snr in snr_list:
            for noise_name, base_noise_wav in noise_chunks:
                
                # ------ A. 生成包含 10 个蛙鸣的纯净轨道 (Grid Placement) ------
                clean_padded = np.zeros(target_length)
                grid_size = target_length // calls_per_chunk # 每个格子的长度 (1秒 = 16000个采样点)
                
                for grid_idx in range(calls_per_chunk):
                    # 随机挑一个树蛙声
                    frog = random.choice(frog_wavs)
                    
                    # 确保蛙鸣能放进 1 秒的格子里
                    max_start_in_grid = grid_size - len(frog)
                    if max_start_in_grid > 0:
                        # 在当前格子里随机找个起始点
                        start = grid_idx * grid_size + random.randint(0, max_start_in_grid)
                        clean_padded[start : start + len(frog)] += frog
                    else:
                        # 极端情况防御：如果蛙鸣居然大于1秒，强行截断放入
                        start = grid_idx * grid_size
                        clean_padded[start : start + grid_size] += frog[:grid_size]
                
                # ------ B. 计算 SNR 并混合缩放 ------
                clean_rms = calculate_rms(clean_padded)
                if clean_rms == 0: clean_rms = 1e-8
                
                noise_rms = calculate_rms(base_noise_wav)
                if noise_rms == 0: noise_rms = 1e-8
                
                # 根据目标 SNR 计算噪声应有的 RMS，并缩放背景噪声
                target_noise_rms = clean_rms / (10 ** (snr / 20))
                scale_factor = target_noise_rms / noise_rms
                scaled_noise = base_noise_wav * scale_factor
                
                # 叠加
                mixed_wav = clean_padded + scaled_noise
                
                # ------ C. 峰值归一化 (防爆音) ------
                max_amp = np.max(np.abs(mixed_wav))
                if max_amp > 0.95:
                    norm_factor = 0.95 / max_amp
                    mixed_wav = mixed_wav * norm_factor
                    clean_padded = clean_padded * norm_factor # 参考音也要等比例缩放，保证映射正确
                
                # ------ D. 保存文件并写入字典 ------
                filename = f"mix_{noise_name}_snr{snr}_{file_idx:04d}.wav"
                out_c_path = os.path.join(out_clean_dir, filename)
                out_n_path = os.path.join(out_noisy_dir, filename)
                
                sf.write(out_c_path, clean_padded, sr)
                sf.write(out_n_path, mixed_wav, sr)
                
                clean_json_dict[filename] = os.path.abspath(out_c_path)
                noisy_json_dict[filename] = os.path.abspath(out_n_path)
                
                file_idx += 1
                pbar.update(1)

    # 4. 输出 JSON 配置文件
    with open(os.path.join(output_dir, "train_clean.json"), "w", encoding="utf-8") as f:
        json.dump(clean_json_dict, f, indent=4)
    with open(os.path.join(output_dir, "train_noisy.json"), "w", encoding="utf-8") as f:
        json.dump(noisy_json_dict, f, indent=4)
        
    print(f"\n生成完毕！共合成 {file_idx} 条数据。")
    print(f"数据保存在: {os.path.abspath(output_dir)}")
    print("可以直接使用 train_clean.json 和 train_noisy.json 进行训练了！")

if __name__ == "__main__":
    main()