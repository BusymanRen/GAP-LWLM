import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from random import randrange
import timm

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def cosine_similarity_loss(H1, H2, eps=1e-1):
    H1r = H1[:, 0, :, :]
    H1i = H1[:, 1, :, :]
    H2r = H2[:, 0, :, :]
    H2i = H2[:, 1, :, :]
    B, M, N = H1r.shape
    xi = H1r.reshape(B, -1)
    xj = H1i.reshape(B, -1)
    yi = H2r.reshape(B, -1)
    yj = H2i.reshape(B, -1)
    xy_dot_r = torch.sum(xi * yi + xj * yj, dim=1)
    xy_dot_i = torch.sum(xi * yj - xj * yi, dim=1)
    norm_x = torch.sqrt(torch.sum(xi ** 2, dim=1) + torch.sum(xj ** 2, dim=1))
    norm_y = torch.sqrt(torch.sum(yi ** 2, dim=1) + torch.sum(yj ** 2, dim=1))
    dot_mod = torch.sqrt(xy_dot_r ** 2 + xy_dot_i ** 2)
    cos_loss = 1 - dot_mod / (norm_x * norm_y + eps)
    return torch.mean(cos_loss)

def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class MAEViTTinyGraphEncoder(nn.Module):
    def __init__(self, output_dim=128, num_graph_tokens=5, unfreeze_ratio=0.8):
        super().__init__()
        self.num_graph_tokens = num_graph_tokens
        
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.backbone.eval()
        
        # Unfreeze Transformer Blocks based on ratio
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            unfreeze_count = int(total_blocks * unfreeze_ratio)
            for i, block in enumerate(self.backbone.blocks):
                if i >= total_blocks - unfreeze_count:
                    for param in block.parameters():
                        param.requires_grad = True
                    block.train()
                else:
                    for param in block.parameters():
                        param.requires_grad = False
        elif unfreeze_ratio > 0:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()
        
        self.proj = nn.Sequential(
            nn.Linear(192, output_dim * num_graph_tokens),
            nn.ReLU()
        )

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        has_grad = any(p.requires_grad for p in self.backbone.parameters())
        if has_grad:
            features = self.backbone(x)
        else:
            with torch.no_grad():
                features = self.backbone(x)
                
        out = self.proj(features)
        B = features.shape[0]
        return out.reshape(B, self.num_graph_tokens, -1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size[0]) * (W // self.patch_size[1])
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x, num_patches


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=512, embed_dim=512, depth=2, num_heads=8, dim_feedforward=2048, dropout=0.1, final_out_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim, bias=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, final_out_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.xavier_normal_(self.mask_token)

    def forward(self, memory, tgt=None, mask_idx=None):
        B, N, _ = memory.shape
        memory_embed = self.latent_to_embed(memory)
        tgt_embed = memory_embed.clone() if tgt is None else tgt
        
        if mask_idx is not None:
            mask_token_embed = self.latent_to_embed(self.mask_token)
            for i in range(B):
                tgt_embed[i, mask_idx[i], :] = mask_token_embed[0, 0, :]
                
        out = self.transformer_decoder(tgt_embed, memory_embed)
        out = self.proj_out(self.norm(out))
        return out


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_scores = self.attn(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        fused = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return fused, attn_weights


class wireless_loc_fm(nn.Module):
    def __init__(
            self, label_dim=567, fshape=4, tshape=4, fstride=4, tstride=4,
            input_fmap=2, embed_dim=512, depth=6, num_heads=8,
            dim_feedforward=2048, dropout=0.1, contrast_out_dim=32,
            BSconf_dim=9, device="cpu", EnvPara=None, graph_unfreeze_ratio=0.8
    ):
        super().__init__()
        self.device = device
        self.EnvPara = EnvPara
        self.BSconf_dim = BSconf_dim
        self.N_graph_tokens = 5
        
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fmap = input_fmap
        self.patch_size = (fshape, tshape)
        
        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=input_fmap, embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_num = 1
        
        self.proj_BSconfig_to_tokens = nn.Sequential(
            nn.Linear(BSconf_dim, int(BSconf_dim * embed_dim / 2)),
            nn.GELU(),
            nn.Linear(int(BSconf_dim * embed_dim / 2), BSconf_dim * embed_dim)
        )
        
        self.mask_graph_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_bs_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_normal_(self.mask_graph_token)
        nn.init.xavier_normal_(self.mask_bs_token)
        
        max_seq_len = 1 + self.N_graph_tokens + self.BSconf_dim + 1024
        pos_embed = get_sinusoid_encoding(max_seq_len, embed_dim)
        self.register_buffer('pos_embed', pos_embed)
        self.embed_dim = embed_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.graph_encoder = MAEViTTinyGraphEncoder(
            output_dim=self.embed_dim,
            num_graph_tokens=self.N_graph_tokens,
            unfreeze_ratio=graph_unfreeze_ratio
        )
        
        if self.EnvPara["task"] in ["SingleBSLoc", "prompt_loc", "inference_SingleBSLoc"]:
            self.SB_positioninglayer = nn.Sequential(
                nn.Linear(self.embed_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
        
        self.decoder_AFMCM_mix = SimpleDecoder(
            latent_dim=EnvPara["AFMCM_embed_dim"], embed_dim=embed_dim, depth=2,
            num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            final_out_dim=fshape * tshape * input_fmap
        )
        self.mask_antenna = nn.Parameter(torch.zeros([1, input_fmap, 1, 1]))
        self.mask_subcarrier = nn.Parameter(torch.zeros([1, input_fmap, 1, 1]))
        nn.init.xavier_normal_(self.mask_antenna)
        nn.init.xavier_normal_(self.mask_subcarrier)
        
        self.decoder_DTI_mix = SimpleDecoder(
            latent_dim=EnvPara["DTI_embed_dim"], embed_dim=embed_dim, depth=2,
            num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            final_out_dim=fshape * tshape * input_fmap
        )
        self.L1loss = nn.L1Loss()
        
        # Configure param freezing at init based on task
        if self.EnvPara["task"] == "prompt_loc":
            self.set_prompt_mode_params(unfreeze_graph_ratio=0.5, unfreeze_transformer_layers=0.0)

    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        return torch.tensor(random.sample(range(0, sequence_len), mask_size))

    def pad_to_divisible(self, x, patch_size):
        _, _, H, W = x.shape
        ph, pw = patch_size
        new_H = ((H + ph - 1) // ph) * ph
        new_W = ((W + pw - 1) // pw) * pw
        pad_h, pad_w = new_H - H, new_W - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (H, W)

    def fm_encoder(self, input, BSconf=None, scene_graphs=None, use_bsconf=True, use_graph=True):
        B = input.shape[0]
        input, orig_size = self.pad_to_divisible(input, self.patch_size)
        
        x_patches, num_patches = self.patch_embed(input)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if scene_graphs is not None and use_graph:
            graph_tokens = self.graph_encoder(scene_graphs)
        else:
            graph_tokens = self.mask_graph_token.expand(B, self.N_graph_tokens, -1)
            
        if BSconf is not None and use_bsconf:
            bs_conf_flat = self.proj_BSconfig_to_tokens(BSconf)
            bs_conf_tokens = bs_conf_flat.view(B, self.BSconf_dim, self.embed_dim)
        else:
            bs_conf_tokens = self.mask_bs_token.expand(B, self.BSconf_dim, -1)

        x = torch.cat((cls_tokens, graph_tokens, bs_conf_tokens, x_patches), dim=1)
        
        total_seq_len = x.size(1)
        x = x + self.pos_embed[:, :total_seq_len, :]
        
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x, num_patches, self.BSconf_dim, orig_size

    def mix_pretraining(self, channel_Antenna_subcarrier_aa, channel_delay_angle_ri, BSconf_all,
                        mask_antenna_ratio=0.25, mask_subcarrier_ratio=0.25,
                        scene_graphs_1=None, scene_graphs_2=None):
        B = channel_Antenna_subcarrier_aa.shape[0]
        x_input = channel_Antenna_subcarrier_aa[:, 0, :]
        x_input_ri = channel_delay_angle_ri[:, 0, :]
        BSconf = BSconf_all[:, 0, :]
        
        DTI_loss_cos = 0
        AFMCM_loss_mse = 0
        
        if self.EnvPara["DTI_flag"] == 1:
            enc_out, num_patches, N_bs_tokens, orig_size = self.fm_encoder(
                x_input, BSconf=BSconf, scene_graphs=scene_graphs_1, use_bsconf=False, use_graph=False
            )
            rec = self.decoder_DTI_mix(enc_out[:, :, :self.EnvPara["DTI_embed_dim"]])
            rec_wo_cls = rec[:, self.cls_token_num + self.N_graph_tokens + N_bs_tokens:, :].transpose(1, 2)
            
            H_orig, W_orig = orig_size
            ph, pw = self.patch_size
            H_padded = ((H_orig + ph - 1) // ph) * ph
            W_padded = ((W_orig + pw - 1) // pw) * pw
            
            fold_layer = nn.Fold(
                output_size=(H_padded, W_padded), kernel_size=(self.fshape, self.tshape),
                stride=(self.fstride, self.tstride)
            ).to(self.device)
            
            x_recon_spatial = fold_layer(rec_wo_cls)[:, :, :H_orig, :W_orig]
            DTI_loss_cos = cosine_similarity_loss(x_input_ri, x_recon_spatial)
        
        if self.EnvPara["AFMCM_flag"] == 1:
            mask_antenna_number = int(x_input.shape[3] * mask_antenna_ratio)
            mask_subcarrier_number = int(x_input.shape[2] * mask_subcarrier_ratio)
            
            mask_dense_ant = torch.ones_like(x_input)
            for i in range(x_input.shape[0]):
                mask_dense_ant[i, :, :, self.gen_maskid_frame(x_input.shape[3], mask_antenna_number)] = 0
            x_masked = x_input * mask_dense_ant + (1 - mask_dense_ant) * self.mask_antenna.expand_as(x_input)
            
            mask_dense_subc = torch.ones_like(x_input)
            for i in range(x_input.shape[0]):
                mask_dense_subc[i, :, self.gen_maskid_frame(x_input.shape[2], mask_subcarrier_number), :] = 0
            x_masked = x_masked * mask_dense_subc + (1 - mask_dense_subc) * self.mask_subcarrier.expand_as(x_input)
            
            enc_out, num_patches, N_bs_tokens, orig_size = self.fm_encoder(
                x_masked, BSconf=BSconf, scene_graphs=scene_graphs_1, use_bsconf=False, use_graph=False
            )
            rec = self.decoder_AFMCM_mix(
                enc_out[:, :, self.EnvPara["DTI_embed_dim"]:self.EnvPara["DTI_embed_dim"] + self.EnvPara["AFMCM_embed_dim"]]
            )
            rec_wo_cls = rec[:, self.cls_token_num + self.N_graph_tokens + N_bs_tokens:, :]
            
            H_orig, W_orig = orig_size
            ph, pw = self.patch_size
            H_padded = ((H_orig + ph - 1) // ph) * ph
            W_padded = ((W_orig + pw - 1) // pw) * pw
            unfold_layer = nn.Unfold(
                kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride)
            ).to(self.device)
            
            x_input_padded = F.pad(x_input, (0, W_padded - W_orig, 0, H_padded - H_orig), mode='constant', value=0)
            input_unfold = unfold_layer(x_input_padded).transpose(1, 2)
            AFMCM_loss_mse = torch.mean((rec_wo_cls - input_unfold) ** 2)
        
        return 20 * DTI_loss_cos + 10 * AFMCM_loss_mse

    def singleBSLoc_with_pilot(self, channel_data, UElocation_all, BSconf_all, scene_graphs=None):
        input = channel_data[:, 0, :]
        y_position = UElocation_all[:, 0, :]
        BSconf = BSconf_all[:, 0, :]
        B, N_s, N_a = input.shape[0], input.shape[2], input.shape[3]
        
        relat_pos = y_position - BSconf[:, :2]
        
        mask_dense_ant_subc = 1 - torch.ones_like(input)
        for i in range(0, N_a, self.EnvPara["pilot_antenna_interval"]):
            for j in range(0, N_s, self.EnvPara["pilot_subcarrier_interval"]):
                mask_dense_ant_subc[:, :, j, i] = 1
        input = input * mask_dense_ant_subc
        
        if self.EnvPara["is_frozen"] == 1:
            modules_to_freeze = [self.patch_embed, self.transformer_encoder, self.norm]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
            self.pos_embed.requires_grad = False
            self.cls_token.requires_grad = False
        
        x, _, _, _ = self.fm_encoder(
            input, BSconf=BSconf, scene_graphs=scene_graphs, use_bsconf=True, use_graph=False
        )
        
        x_mean = x[:, 0, :].view((B, -1))
        pred = self.SB_positioninglayer(x_mean)
        
        dis = torch.sum(25 * (pred - relat_pos) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))
        return pred, mse

    def prompt_loc_with_multimodal(self, channel_data, UElocation_all, BSconf_all, scene_graphs=None):
        input = channel_data[:, 0, :]
        y_position = UElocation_all[:, 0, :]
        BSconf = BSconf_all[:, 0, :]
        B = input.shape[0]
        
        relat_pos = y_position - BSconf[:, :2]
        
        mask_dense_ant_subc = 1 - torch.ones_like(input)
        if "pilot_antenna_interval" in self.EnvPara:
            for i in range(0, input.shape[3], self.EnvPara["pilot_antenna_interval"]):
                for j in range(0, input.shape[2], self.EnvPara["pilot_subcarrier_interval"]):
                    mask_dense_ant_subc[:, :, j, i] = 1
            input = input * mask_dense_ant_subc
        
        x, _, _, _ = self.fm_encoder(
            input, BSconf=BSconf, scene_graphs=scene_graphs, use_bsconf=True, use_graph=True
        )
        
        x_mean = x[:, 0, :].view((B, -1))
        pred = self.SB_positioninglayer(x_mean)
        
        dis = torch.sum(25 * (pred - relat_pos) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))
        return pred, mse
    
    def set_prompt_mode_params(self, unfreeze_graph_ratio=0.8, unfreeze_transformer_layers=1.0):
        if hasattr(self.graph_encoder.backbone, 'blocks'):
            total_blocks = len(self.graph_encoder.backbone.blocks)
            unfreeze_count = int(total_blocks * unfreeze_graph_ratio)
            for i, block in enumerate(self.graph_encoder.backbone.blocks):
                for param in block.parameters():
                    param.requires_grad = (i >= total_blocks - unfreeze_count)
        
        total_enc_layers = len(self.transformer_encoder.layers)
        unfreeze_enc_count = int(total_enc_layers * unfreeze_transformer_layers)
        for i, layer in enumerate(self.transformer_encoder.layers):
            for param in layer.parameters():
                param.requires_grad = (i >= total_enc_layers - unfreeze_enc_count)
                
        for param in self.patch_embed.parameters():
            param.requires_grad = False
            
        for param in self.norm.parameters():
            param.requires_grad = False
        
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = True
        self.proj_BSconfig_to_tokens.requires_grad_(False)
        self.SB_positioninglayer.requires_grad_(True)
        self.mask_graph_token.requires_grad = False
        self.mask_bs_token.requires_grad = False
    
    def forward(self, channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri,
                UElocation_all, BSconf_all, task="pretrain_mix", scene_graphs=None, cluster=True):
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.transpose(3, 4)
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.transpose(3, 4)
        channel_delay_angle_ri = channel_delay_angle_ri.transpose(3, 4)
        UElocation_all = UElocation_all[:, :, :2]
        
        if task == "SingleBSLoc":
            _, mse = self.singleBSLoc_with_pilot(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, scene_graphs=scene_graphs)
            return mse
        elif task == "inference_SingleBSLoc":
            return self.singleBSLoc_with_pilot(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, scene_graphs=scene_graphs)
        elif task == "prompt_loc":
            _, mse = self.prompt_loc_with_multimodal(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, scene_graphs=scene_graphs)
            return mse
        elif task == "pretrain_mix":
            scene_graph_1 = scene_graphs[:, 0, :, :, :] if scene_graphs is not None else None
            scene_graph_2 = scene_graphs[:, 1, :, :, :] if scene_graphs is not None else None
            return self.mix_pretraining(
                channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, BSconf_all=BSconf_all,
                scene_graphs_1=scene_graph_1, scene_graphs_2=scene_graph_2
            )
        else:
            raise ValueError(f"Unsupported task: {task}")