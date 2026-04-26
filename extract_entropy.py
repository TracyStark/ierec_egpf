# extract_entropy_ratios.py
import torch
import pickle
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, ensure_dir

# 1. 加载配置（使用你当前的 NEW.yaml）
config = Config(model='NEW', dataset='ml-100k', config_file_list=['recbole/properties/model/NEW.yaml'])
init_logger(config)
ensure_dir(config['checkpoint_dir'])

# 2. 加载数据集和模型
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
try:
    checkpoint = torch.load(
        'saved/NEW-Apr-23-2026_18-14-26.pth',
        map_location=config['device'],
        weights_only=True,
    )  # ← 请确认你的pth文件名
except (TypeError, pickle.UnpicklingError):
    checkpoint = torch.load('saved/NEW-Apr-23-2026_18-14-26.pth', map_location=config['device'])
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 3. 计算所有序列的 Interest Entropy 并统计比例（τ=0.3）
with torch.no_grad():
    entropy_list = []
    for batch in train_data:  # 使用训练集全量数据
        item_seq = batch[model.ITEM_SEQ]
        item_seq_len = batch[model.ITEM_SEQ_LEN]
        entropy = model.interest_entropy(item_seq, item_seq_len, theta=model.theta)
        entropy_list.extend(entropy.cpu().numpy())

    entropy_tensor = torch.tensor(entropy_list)
    low_thresh = torch.quantile(entropy_tensor, 0.3)
    high_thresh = torch.quantile(entropy_tensor, 0.7)

    low_ratio = (entropy_tensor < low_thresh).float().mean().item() * 100
    medium_ratio = ((entropy_tensor >= low_thresh) & (entropy_tensor <= high_thresh)).float().mean().item() * 100
    high_ratio = (entropy_tensor > high_thresh).float().mean().item() * 100

print("="*50)
print("Interest Entropy 序列比例统计 (τ=0.3)")
print(f"Low-entropy ratio  : {low_ratio:.2f}%")
print(f"Medium-entropy ratio: {medium_ratio:.2f}%")
print(f"High-entropy ratio : {high_ratio:.2f}%")
print("="*50)