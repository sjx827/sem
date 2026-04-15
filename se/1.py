import json
import random

# 1. 读取之前生成的字典格式 JSON
with open("data/train_clean.json", "r", encoding="utf-8") as f:
    clean_dict = json.load(f)
with open("data/train_noisy.json", "r", encoding="utf-8") as f:
    noisy_dict = json.load(f)

# 2. 提取所有键并打乱顺序
keys = list(clean_dict.keys())
random.seed(1234)
random.shuffle(keys)

# 3. 按打乱后的顺序转换为纯路径列表 (List)
clean_list = [clean_dict[k] for k in keys]
noisy_list = [noisy_dict[k] for k in keys]

# 4. 划分：前 2800 条用于训练，后 200 条用于验证监控
train_clean = clean_list[:-200]
train_noisy = noisy_list[:-200]
valid_clean = clean_list[-200:]
valid_noisy = noisy_list[-200:]

# 5. 保存到 SEMamba 配置默认读取的 data 目录下
with open("data/train_clean.json", "w", encoding="utf-8") as f:
    json.dump(train_clean, f, indent=4)
with open("data/train_noisy.json", "w", encoding="utf-8") as f:
    json.dump(train_noisy, f, indent=4)
with open("data/valid_clean.json", "w", encoding="utf-8") as f:
    json.dump(valid_clean, f, indent=4)
with open("data/valid_noisy.json", "w", encoding="utf-8") as f:
    json.dump(valid_noisy, f, indent=4)

print("格式转换完毕！训练集和验证集已就绪。")