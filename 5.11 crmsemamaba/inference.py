import os
import argparse
import torch
import librosa
import soundfile as sf
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from utils.util import load_config

def calculate_sisdr(reference, estimated):
    """计算 SI-SDR 指标 (消除幅度缩放影响)"""
    eps = 1e-8
    reference = reference - torch.mean(reference)
    estimated = estimated - torch.mean(estimated)
    
    alpha = (torch.sum(reference * estimated) + eps) / (torch.sum(reference ** 2) + eps)
    target = alpha * reference
    noise = estimated - target
    
    sisdr = 10 * torch.log10((torch.sum(target ** 2) + eps) / (torch.sum(noise ** 2) + eps))
    return sisdr.item()

def calculate_snr(reference, estimated):
    """计算标准 SNR 指标"""
    eps = 1e-8
    noise = reference - estimated
    snr = 10 * torch.log10(torch.sum(reference**2) / (torch.sum(noise**2) + eps))
    return snr.item()

def inference(args, device):
    # 1. 加载配置
    cfg = load_config(args.config)
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']

    # 2. 初始化你的 CRM SEMamba 模型
    print(">>> 正在加载生成器模型...")
    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])
    model.eval()

    os.makedirs(args.output_folder, exist_ok=True)
    print(f">>> 降噪音频将保存至: {os.path.abspath(args.output_folder)}\n")

    total_sisdr, total_snr, file_count = 0.0, 0.0, 0

    # 打印优美的表头
    print(f"{'处理状态':<12} | {'文件名 (Filename)':<40} | {'SI-SDR (dB)':<12} | {'SNR (dB)':<12}")
    print("-" * 85)

    with torch.no_grad():
        for fname in sorted(os.listdir(args.input_folder)):
            if not fname.lower().endswith('.wav'):
                continue
            
            # --- 前向降噪处理 ---
            noisy_path = os.path.join(args.input_folder, fname)
            noisy_wav, _ = librosa.load(noisy_path, sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            # 物理量级归一化
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav_norm = (noisy_wav * norm_factor).unsqueeze(0)

            # STFT -> CRM 掩蔽 -> ISTFT
            noisy_amp, noisy_pha, _ = mag_phase_stft(noisy_wav_norm, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            
            # 恢复真实的物理音量
            audio_g = audio_g.squeeze() / norm_factor  
            
            # 保存增强后的音频
            output_file = os.path.join(args.output_folder, fname)
            sf.write(output_file, audio_g.cpu().numpy(), sampling_rate, 'PCM_16')

            # --- 指标计算逻辑 ---
            status = "[保存成功]"
            sisdr_str = "N/A"
            snr_str = "N/A"

            if args.clean_folder:
                clean_path = os.path.join(args.clean_folder, fname)
                if os.path.exists(clean_path):
                    clean_wav, _ = librosa.load(clean_path, sr=sampling_rate)
                    clean_wav = torch.FloatTensor(clean_wav).to(device)
                    
                    # 严谨截断，防止由于 STFT 补零导致的长度微小不一致
                    min_len = min(len(clean_wav), len(audio_g))
                    c_ref = clean_wav[:min_len]
                    a_est = audio_g[:min_len]

                    # 计算跑分
                    sisdr_val = calculate_sisdr(c_ref, a_est)
                    snr_val = calculate_snr(c_ref, a_est)
                    
                    total_sisdr += sisdr_val
                    total_snr += snr_val
                    file_count += 1
                    
                    status = "[指标计算]"
                    sisdr_str = f"{sisdr_val:.2f}"
                    snr_str = f"{snr_val:.2f}"
                else:
                    status = "[缺失参考]" # 找不到纯净文件，但已经降噪并保存
            
            # 截断过长的文件名以保持表格整洁
            display_name = fname if len(fname) <= 38 else fname[:35] + "..."
            print(f"{status:<12} | {display_name:<40} | {sisdr_str:>12} | {snr_str:>12}")

    # --- 最终汇总 ---
    print("-" * 85)
    if file_count > 0:
        print(f">>> 评估汇总 ({file_count} 个文件): 平均 SI-SDR = {total_sisdr/file_count:.2f} dB, 平均 SNR = {total_snr/file_count:.2f} dB")
    else:
        print(">>> 提示: 未匹配到任何参考音频，未计算平均指标。(若需计算，请检查 clean_folder 路径及文件名是否一致)")
    print(">>> 推理任务全部完成！\n")


def main():
    parser = argparse.ArgumentParser(description="SEMamba CRM Inference Script")
    parser.add_argument('--input_folder', required=True, help='带噪音频所在目录')
    parser.add_argument('--clean_folder', default=None, help='纯净参考音频所在目录 (可选，用于计算指标)')
    parser.add_argument('--output_folder', default='results', help='降噪结果输出目录 (默认: results)')
    parser.add_argument('--config', required=True, help='YAML 配置文件路径')
    parser.add_argument('--checkpoint_file', required=True, help='.pth 权重文件路径')
    args = parser.parse_args()

    # 硬件检测
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("警告: 未检测到 GPU，将使用 CPU 进行极慢速推理...")
        device = torch.device('cpu')

    inference(args, device)

if __name__ == '__main__':
    main()