import os
import re
import random
import h5py
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset


# ====================== 工具函数 ======================
def channel_normalization(H):
    P_current = np.sum(np.abs(H) ** 2)
    alpha = np.sqrt((H.shape[2]) / P_current)
    return H * alpha


def freq_to_delay_angle_domain(H_freq):
    H_complex = H_freq[0] + 1j * H_freq[1]
    H_delay = np.fft.ifft(H_complex, axis=0)
    H_delay_angle_complex = np.fft.fft(H_delay, axis=1) / np.sqrt(H_delay.shape[0])
    return H_delay_angle_complex


# ====================== GroupedBatchSampler ======================
class GroupedBatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.config_to_indices = defaultdict(list)
        
        # 优化：直接通过 idx_to_config 属性获取映射，避免对 Dataset 进行全量磁盘 I/O 采样
        for idx in range(len(dataset)):
            if hasattr(dataset, 'idx_to_config'):
                config_key = dataset.idx_to_config[idx]
            else:
                sample = dataset[idx]  # fallback
                config_key = sample[-2]
            self.config_to_indices[config_key].append(idx)

    def __iter__(self):
        all_batches = []
        for indices in self.config_to_indices.values():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if batch:
                    all_batches.append(batch)
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for indices in self.config_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


# ====================== 预训练数据集（双基站配对） ======================
class PretrainSceneDataset(Dataset):
    def __init__(self, EnvPara, is_val=0):
        self.EnvPara = EnvPara
        self.input_fmap = EnvPara["input_fmap"]
        self.is_val = is_val
        self.data_root = EnvPara["data_root"]
        self.len = EnvPara.get("FT_dataset", 10000)
        self.scene_ids = EnvPara.get("scene_ids", list(range(1, 3)))
        
        # 内存缓存，防止跨进程下重复加载静态场景图导致 I/O 拥塞
        self.scene_graph_cache = {} 

        self.scene_pairs = defaultdict(list)
        for scene_id in self.scene_ids:
            scene_path = os.path.join(self.data_root, f"scene_{scene_id}")
            if not os.path.exists(scene_path):
                continue

            try:
                if self.is_val:
                    idx_path = os.path.join(scene_path, "pretrain", "val_indices.npy")
                else:
                    idx_path = os.path.join(scene_path, "pretrain","train.npy")
                indices = np.load(idx_path)[:self.len]
            except Exception:
                continue

            pattern = re.compile(r"BS(\d+)_(\d+)sc_(\d+)at_(\d+)M")
            config_to_bs = defaultdict(list)
            for folder in os.listdir(scene_path):
                match = pattern.match(folder)
                if match:
                    bs_id = int(match.group(1))
                    sub = int(match.group(2))
                    ant = int(match.group(3))
                    bw = int(match.group(4))
                    config_to_bs[(sub, ant)].append((bs_id, bw))

            for (sub, ant), bs_list in config_to_bs.items():
                if len(bs_list) == 1:
                    (bs1, bw1) = bs_list[0]
                    (bs2, bw2) = (bs1, bw1)
                elif len(bs_list) >= 2:
                    (bs1, bw1), (bs2, bw2) = bs_list[0], bs_list[1]
                else:
                    continue
                self.scene_pairs[(sub, ant)].append((scene_id, [bs1, bs2], [bw1, bw2], indices))

        if not self.scene_pairs:
            raise ValueError(f"No valid scene pairs found in {self.scene_ids}")

        valid_configs = list(self.scene_pairs.keys())
        self.idx_to_config = [random.choice(valid_configs) for _ in range(self.len)]

    def load_graph_cached(self, scene_id, bs, sub, ant, bw):
        cache_key = (scene_id, bs, sub, ant, bw)
        if cache_key not in self.scene_graph_cache:
            scene_path = os.path.join(self.data_root, f"scene_{scene_id}")
            folder = f"BS{bs}_{sub}sc_{ant}at_{bw}M"
            
            rss_path = os.path.join(scene_path, folder, "rss.npy")
            los_path = os.path.join(scene_path, folder, "los_probability_heatmap.npy")
            building_path = os.path.join(scene_path, f"scene_{scene_id}.npy")
            
            rss = np.load(rss_path)
            los = np.load(los_path)
            building = np.load(building_path)
            
            rss_norm = (rss - rss.min()) / (rss.max() - rss.min() + 1e-8)
            los_norm = np.clip(los, 0.0, 1.0)
            building_norm = building.astype(np.float32) / 10.0
            self.scene_graph_cache[cache_key] = np.stack([rss_norm, los_norm, building_norm], axis=0)
            
        return self.scene_graph_cache[cache_key]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 优化：消除递归导致的 Stack Overflow，使用带有上限的重试循环
        for retry in range(20):
            try:
                chosen_sub, chosen_ant = self.idx_to_config[idx]
                candidates = self.scene_pairs[(chosen_sub, chosen_ant)]
                scene_id, [bs1, bs2], [bw1, bw2], all_indices = random.choice(candidates)
                sample_idx = random.choice(all_indices)
                scene_path = os.path.join(self.data_root, f"scene_{scene_id}")

                file1 = os.path.join(scene_path, f"BS{bs1}_{chosen_sub}sc_{chosen_ant}at_{bw1}M", f"{bs1}_{sample_idx}.h5py")
                file2 = os.path.join(scene_path, f"BS{bs2}_{chosen_sub}sc_{chosen_ant}at_{bw2}M", f"{bs2}_{sample_idx}.h5py")

                def load_channel_and_meta(filepath):
                    with h5py.File(filepath, 'r') as f:
                        ch = f["channel_real"][:] + 1j * f["channel_imag"][:]
                        # 优化：去除了冗余维度切片
                        UE = f["UElocation"][:2].flatten()
                        BS = f["BSlocation"][:2].flatten()
                        dist = float(f["distance"][()])
                        angle = float(f["angle"][()][0])
                        sub_num = float(f["sub_num"][()])
                        carrier_fre = float(f["carrier_frequency"][()])
                        tx_ant = f["tx_antenna"][()]
                        tx_ori = f["tx_orientation"][()]
                        ori = (float(tx_ori[0][0]), float(tx_ori[1][0]), float(tx_ori[2][0]))
                    return ch, UE, BS, dist, angle, sub_num, carrier_fre, tx_ant, ori

                ch1, UE1, BS1, d1, a1, sub1, cf1, ta1, ori1 = load_channel_and_meta(file1)
                ch2, UE2, BS2, d2, a2, sub2, cf2, ta2, ori2 = load_channel_and_meta(file2)

                if np.sum(np.abs(ch1) ** 2) <= 0 or np.sum(np.abs(ch2) ** 2) <= 0:
                    idx = (idx + 1) % self.len
                    continue

                ch1 = channel_normalization(ch1)
                ch2 = channel_normalization(ch2)

                def build_repr(ch, ant, sub):
                    aa = np.zeros([self.input_fmap, ant, sub], dtype=np.float32)
                    ri = np.zeros_like(aa)
                    aa[0] = np.abs(ch).astype(np.float32)
                    aa[1] = np.angle(ch).astype(np.float32)
                    ri[0] = np.real(ch).astype(np.float32)
                    ri[1] = np.imag(ch).astype(np.float32)
                    da_complex = freq_to_delay_angle_domain(ri)
                    da_ri = np.zeros_like(ri)
                    da_ri[0] = np.real(da_complex).astype(np.float32)
                    da_ri[1] = np.imag(da_complex).astype(np.float32)
                    return aa, ri, da_ri

                aa1, ri1, da1 = build_repr(ch1, chosen_ant, chosen_sub)
                aa2, ri2, da2 = build_repr(ch2, chosen_ant, chosen_sub)

                channel_aa = np.zeros((2, self.input_fmap, chosen_ant, chosen_sub), dtype=np.float32)
                channel_ri = np.zeros_like(channel_aa)
                channel_da = np.zeros_like(channel_aa)
                UElocation_all = np.zeros((2, 4), dtype=np.float32)
                BSconf_all = np.zeros((2, 10), dtype=np.float32)

                channel_aa[0] = aa1
                channel_aa[1] = aa2
                channel_ri[0] = ri1
                channel_ri[1] = ri2
                channel_da[0] = da1
                channel_da[1] = da2

                UElocation_all[0, :2] = (UE1 + 16) / 32
                UElocation_all[1, :2] = (UE2 + 16) / 32
                UElocation_all[0, 2] = d1 / 100
                UElocation_all[1, 2] = d2 / 100
                UElocation_all[0, 3] = a1 / 100
                UElocation_all[1, 3] = a2 / 100

                def make_bsconf(BSloc, bw, sub_num, carrier_fre, tx_ant, ori):
                    conf = np.zeros(10, dtype=np.float32)
                    conf[:2] = (BSloc + 16) / 32
                    conf[2] = bw / 10
                    conf[3] = sub_num / 100
                    conf[4] = carrier_fre / 100
                    conf[5] = tx_ant[0] / 100
                    conf[6] = tx_ant[1] / 100
                    conf[7] = ori[0] / 10
                    conf[8] = ori[1] / 10
                    conf[9] = ori[2] / 10
                    return conf

                BSconf_all[0] = make_bsconf(BS1, bw1, sub1, cf1, ta1, ori1)
                BSconf_all[1] = make_bsconf(BS2, bw2, sub2, cf2, ta2, ori2)

                # 使用缓存加载场景图
                graph1 = self.load_graph_cached(scene_id, bs1, chosen_sub, chosen_ant, bw1)
                graph2 = self.load_graph_cached(scene_id, bs2, chosen_sub, chosen_ant, bw2)
                
                scene_graph = np.stack([graph1, graph2], axis=0) 
                config_key = (chosen_sub, chosen_ant)
                
                return (
                    torch.from_numpy(channel_aa).float(),
                    torch.from_numpy(channel_ri).float(),
                    torch.from_numpy(channel_da).float(),
                    torch.from_numpy(UElocation_all).float(),
                    torch.from_numpy(BSconf_all).float(),
                    config_key,
                    torch.from_numpy(scene_graph).float()
                )
            except Exception:
                idx = (idx + 1) % self.len
                
        raise RuntimeError("Failed to load a valid pretrain sample after multiple retries. Check dataset integrity.")


# ====================== 微调数据集（单基站） ======================
class FinetuneSceneDataset(Dataset):
    def __init__(self, EnvPara, is_val=0):
        self.EnvPara = EnvPara
        self.input_fmap = EnvPara["input_fmap"]
        self.is_val = is_val
        self.data_root = EnvPara["data_root"]
        self.len = EnvPara.get("FT_dataset", 6000)
        default_scenes = [1]
        self.scene_ids = EnvPara.get("scene_ids", default_scenes)

        self.scene_graph_cache = {}

        self.scene_config_map = defaultdict(list)
        for scene_id in self.scene_ids:
            scene_path = os.path.join(self.data_root, f"scene_{scene_id}")
            if not os.path.exists(scene_path):
                continue

            try:
                if self.is_val:
                    idx_path = os.path.join(scene_path, "tune","val_indices.npy")
                else:
                    idx_path = os.path.join(scene_path, "tune","train.npy")
                indices = np.load(idx_path)[:self.len]
            except Exception:
                continue

            pattern = re.compile(r"BS(\d+)_(\d+)sc_(\d+)at_(\d+)M")
            for folder in os.listdir(scene_path):
                match = pattern.match(folder)
                if match:
                    bs_id = int(match.group(1))
                    sub = int(match.group(2))
                    ant = int(match.group(3))
                    bw = int(match.group(4))
                    self.scene_config_map[(sub, ant)].append((scene_id, bs_id, bw, indices))

        if not self.scene_config_map:
            raise ValueError(f"No valid single-BS configs found in {self.scene_ids}")

        valid_configs = list(self.scene_config_map.keys())
        self.idx_to_config = [random.choice(valid_configs) for _ in range(self.len)]

    def load_graph_cached(self, scene_id, bs, sub, ant, bw):
        cache_key = (scene_id, bs, sub, ant, bw)
        if cache_key not in self.scene_graph_cache:
            scene_path = os.path.join(self.data_root, f"scene_{scene_id}")
            folder = f"BS{bs}_{sub}sc_{ant}at_{bw}M"
            
            rss_path = os.path.join(scene_path, folder, "rss.npy")
            los_path = os.path.join(scene_path, folder, "los_probability_heatmap.npy")
            building_path = os.path.join(scene_path, f"scene_{scene_id}.npy")
            
            rss = np.load(rss_path)
            los = np.load(los_path)
            building = np.load(building_path)
            
            rss_norm = (rss - rss.min()) / (rss.max() - rss.min() + 1e-8)
            los_norm = np.clip(los, 0.0, 1.0)
            building_norm = building.astype(np.float32) / 10.0
            self.scene_graph_cache[cache_key] = np.stack([rss_norm, los_norm, building_norm], axis=0)
            
        return self.scene_graph_cache[cache_key]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        for retry in range(20):
            try:
                chosen_sub, chosen_ant = self.idx_to_config[idx]
                candidates = self.scene_config_map.get((chosen_sub, chosen_ant))
                if not candidates:
                    idx = (idx + 1) % self.len
                    continue

                scene_id, bs_id, bw, all_indices = random.choice(candidates)
                sample_idx = random.choice(all_indices)
                scene_path = os.path.join(self.data_root, f"scene_{scene_id}")

                folder = f"BS{bs_id}_{chosen_sub}sc_{chosen_ant}at_{bw}M"
                h5_path = os.path.join(scene_path, folder, f"{bs_id}_{sample_idx}.h5py")

                with h5py.File(h5_path, 'r') as f:
                    ch = f["channel_real"][:] + 1j * f["channel_imag"][:]
                    UE = f["UElocation"][:2].flatten()
                    BS = f["BSlocation"][:2].flatten()
                    dist = float(f["distance"][()])
                    angle = float(f["angle"][()][0])
                    sub_num = float(f["sub_num"][()])
                    carrier_fre = float(f["carrier_frequency"][()])
                    
                    # 修复 Bug: 把 f["tx_orientation"] 更正为 f["tx_antenna"]
                    tx_ant = f["tx_antenna"][()]
                    
                    tx_ori = f["tx_orientation"][()]
                    ori = (float(tx_ori[0][0]), float(tx_ori[1][0]), float(tx_ori[2][0]))

                total_elements = ch.size
                zero_elements = np.sum(np.abs(ch) <= 1e-12)
                if np.sum(np.abs(ch) ** 2) <= 0 or (zero_elements / total_elements > 0.1):
                    idx = (idx + 1) % self.len
                    continue

                ch = channel_normalization(ch)

                aa = np.zeros([self.input_fmap, chosen_ant, chosen_sub], dtype=np.float32)
                ri = np.zeros_like(aa)
                aa[0] = np.abs(ch).astype(np.float32)
                aa[1] = np.angle(ch).astype(np.float32)
                ri[0] = np.real(ch).astype(np.float32)
                ri[1] = np.imag(ch).astype(np.float32)
                da_complex = freq_to_delay_angle_domain(ri)
                da_ri = np.zeros_like(ri)
                da_ri[0] = np.real(da_complex).astype(np.float32)
                da_ri[1] = np.imag(da_complex).astype(np.float32)

                # 使用缓存加载场景图
                scene_graph = self.load_graph_cached(scene_id, bs_id, chosen_sub, chosen_ant, bw)

                channel_aa = np.zeros((1, self.input_fmap, chosen_ant, chosen_sub), dtype=np.float32)
                channel_ri = np.zeros_like(channel_aa)
                channel_da = np.zeros_like(channel_aa)
                channel_aa[0] = aa
                channel_ri[0] = ri
                channel_da[0] = da_ri

                UElocation_all = np.zeros((1, 4), dtype=np.float32)
                BSconf_all = np.zeros((1, 9), dtype=np.float32)   
                
                UElocation_all[0, :2] = (UE + 16) / 32
                UElocation_all[0, 2] = dist / 100
                UElocation_all[0, 3] = angle / 100

                BSconf_all[0, :2] = (BS + 16) / 32
                BSconf_all[0, 2] = bw / 10
                BSconf_all[0, 3] = sub_num / 100
                BSconf_all[0, 4] = carrier_fre / 100
                BSconf_all[0, 5] = tx_ant[1] / 100  
                BSconf_all[0, 6] = ori[0] / 10
                BSconf_all[0, 7] = ori[1] / 10
                BSconf_all[0, 8] = ori[2] / 10

                config_key = (chosen_sub, chosen_ant)
                
                return (
                    torch.from_numpy(channel_aa).float(),
                    torch.from_numpy(channel_ri).float(),
                    torch.from_numpy(channel_da).float(),
                    torch.from_numpy(UElocation_all).float(),
                    torch.from_numpy(BSconf_all).float(),
                    config_key,
                    torch.from_numpy(scene_graph).float()
                )
            except Exception:
                idx = (idx + 1) % self.len
                
        raise RuntimeError("Failed to load a valid finetune sample after multiple retries.")


# ====================== 测试数据集 ======================
class TestSceneDataset(Dataset):
    def __init__(self, EnvPara, scene_id=127, target_bs_list=None):
        self.EnvPara = EnvPara
        self.input_fmap = EnvPara["input_fmap"]
        self.data_root = EnvPara["data_root"]
        self.scene_id = scene_id
        self.target_bs_list = target_bs_list or [1]

        self.scene_graph_cache = {}

        self.scene_path = os.path.join(self.data_root, f"scene_{scene_id}")
        self.valid_combinations = self._scan_valid_combinations()
        if not self.valid_combinations:
            raise ValueError(f"No valid data in {self.scene_path}")

        idx_path = os.path.join(self.scene_path, "tune", "test_indices.npy")
        self.data_list = np.load(idx_path)
        self.len = len(self.data_list)

    def _scan_valid_combinations(self):
        pattern = re.compile(r"BS(\d+)_(\d+)sc_(\d+)at_(\d+)M")
        combos = []
        for folder in os.listdir(self.scene_path):
            match = pattern.match(folder)
            if match:
                combos.append((int(match.group(1)), int(match.group(2)),
                               int(match.group(3)), int(match.group(4))))
        return combos

    def load_graph_cached(self, folder, use_heatmap64=True):
        if folder not in self.scene_graph_cache:
            rss_path = os.path.join(self.scene_path, folder, "rss.npy")
            # 保持测试集使用 heatmap64 的原始逻辑
            los_name = "los_probability_heatmap64.npy" if use_heatmap64 else "los_probability_heatmap.npy"
            los_path = os.path.join(self.scene_path, folder, los_name)
            building_path = os.path.join(self.scene_path, f"scene_{self.scene_id}.npy")
            
            rss = np.load(rss_path)
            los = np.load(los_path)
            building = np.load(building_path)
            
            rss_norm = (rss - rss.min()) / (rss.max() - rss.min() + 1e-8)
            los_norm = np.clip(los, 0.0, 1.0)
            building_norm = building.astype(np.float32) / 10.0
            self.scene_graph_cache[folder] = np.stack([rss_norm, los_norm, building_norm], axis=0)
            
        return self.scene_graph_cache[folder]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        for retry in range(20):
            try:
                BS_Num = self.EnvPara.get("BS_Num", 4)
                if not hasattr(self, '_fixed_dims'):
                    for bs, sub, ant, bw in self.valid_combinations:
                        if bs in self.target_bs_list:
                            self._fixed_dims = (ant, sub)
                            break
                    else:
                        self._fixed_dims = (16, 100)  

                fixed_ant, fixed_sub = self._fixed_dims

                channel_aa = np.zeros((BS_Num, self.input_fmap, fixed_ant, fixed_sub), dtype=np.float32)
                channel_ri = np.zeros_like(channel_aa)
                channel_da = np.zeros_like(channel_aa)
                UElocation_all = np.zeros((BS_Num, 4), dtype=np.float32)
                BSconf_all = np.zeros((BS_Num, 9), dtype=np.float32)

                loaded_bs_count = 0
                scene_graph = None

                for bs_id in self.target_bs_list:
                    if loaded_bs_count >= BS_Num:
                        break
                    configs = [(sub, ant, bw) for (bs, sub, ant, bw) in self.valid_combinations if bs == bs_id]
                    if not configs:
                        continue
                        
                    sub, ant, bw = random.choice(configs)
                    folder = f"BS{bs_id}_{sub}sc_{ant}at_{bw}M"
                    h5_path = os.path.join(self.scene_path, folder, f"{bs_id}_{self.data_list[idx]}.h5py")

                    with h5py.File(h5_path, 'r') as f:
                        ch = f["channel_real"][:] + 1j * f["channel_imag"][:]
                        if np.sum(np.abs(ch) ** 2) <= 0:
                            raise ValueError("empty")
                        
                        UE = f["UElocation"][:2].flatten()
                        if UE[0] < -12.5 and UE[1] < -12.5:
                            raise ValueError("filtered")
                            
                        BS = f["BSlocation"][:2].flatten()
                        dist = float(f["distance"][()])
                        angle = float(f["angle"][()][0])
                        sub_num = float(f["sub_num"][()])
                        carrier_fre = float(f["carrier_frequency"][()])
                        tx_ant = f["tx_antenna"][()]
                        tx_ori = f["tx_orientation"][()]
                        ori = (float(tx_ori[0][0]), float(tx_ori[1][0]), float(tx_ori[2][0]))

                    ch = channel_normalization(ch)
                    aa = np.zeros([self.input_fmap, ant, sub], dtype=np.float32)
                    ri = np.zeros_like(aa)
                    aa[0] = np.abs(ch).astype(np.float32)
                    aa[1] = np.angle(ch).astype(np.float32)
                    ri[0] = np.real(ch).astype(np.float32)
                    ri[1] = np.imag(ch).astype(np.float32)
                    da_complex = freq_to_delay_angle_domain(ri)
                    da_ri = np.zeros_like(ri)
                    da_ri[0] = np.real(da_complex).astype(np.float32)
                    da_ri[1] = np.imag(da_complex).astype(np.float32)

                    channel_aa[loaded_bs_count] = aa
                    channel_ri[loaded_bs_count] = ri
                    channel_da[loaded_bs_count] = da_ri
                    UElocation_all[loaded_bs_count, :2] = (UE + 16) / 32
                    UElocation_all[loaded_bs_count, 2] = dist / 100
                    UElocation_all[loaded_bs_count, 3] = angle / 100
                    BSconf_all[loaded_bs_count, :2] = (BS + 16) / 32
                    BSconf_all[loaded_bs_count, 2] = bw / 10
                    BSconf_all[loaded_bs_count, 3] = sub_num / 100
                    BSconf_all[loaded_bs_count, 4] = carrier_fre / 100
                    BSconf_all[loaded_bs_count, 5] = tx_ant[1] / 100
                    BSconf_all[loaded_bs_count, 6] = ori[0] / 10
                    BSconf_all[loaded_bs_count, 7] = ori[1] / 10
                    BSconf_all[loaded_bs_count, 8] = ori[2] / 10
                    loaded_bs_count += 1

                    if scene_graph is None:
                        scene_graph = self.load_graph_cached(folder, use_heatmap64=True)

                if loaded_bs_count == 0:
                    raise ValueError("No BS loaded")

                config_key = (fixed_sub, fixed_ant)
                return (
                    torch.from_numpy(channel_aa).float(),
                    torch.from_numpy(channel_ri).float(),
                    torch.from_numpy(channel_da).float(),
                    torch.from_numpy(UElocation_all).float(),
                    torch.from_numpy(BSconf_all).float(),
                    config_key,
                    torch.from_numpy(scene_graph).float()
                )

            except Exception:
                idx = (idx + 1) % self.len
                
        raise RuntimeError("Failed to load a valid test sample after multiple retries.")