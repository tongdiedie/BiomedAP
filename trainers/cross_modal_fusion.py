"""
Cross-Modal Prompt Fusion Module
双向注意力机制实现Visual-Text Prompt交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalPromptFusion(nn.Module):
    """
    双向注意力Prompt融合模块
    
    让Visual Prompt和Text Prompt在编码过程中相互增强:
    - Visual → Text: Visual Prompt关注Text Prompt的语义信息
    - Text → Visual: Text Prompt关注Visual Prompt的视觉特征
    """
    def __init__(self, visual_dim=768, text_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        
        # ========== 【关键修改】投影到统一维度 ==========
        # 选择较大的维度作为统一维度
        self.unified_dim = max(visual_dim, text_dim)

        # 投影层：将不同维度映射到统一维度
        self.visual_proj = nn.Linear(visual_dim, self.unified_dim) if visual_dim != self.unified_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, self.unified_dim) if text_dim != self.unified_dim else nn.Identity()
        
        # 反投影层：映射回原始维度
        self.visual_out_proj = nn.Linear(self.unified_dim, visual_dim) if visual_dim != self.unified_dim else nn.Identity()
        self.text_out_proj = nn.Linear(self.unified_dim, text_dim) if text_dim != self.unified_dim else nn.Identity()

        # Visual关注Text的多头注意力
        self.visual_to_text_attn = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # 使用(seq_len, batch, dim)格式
        )
        
        # Text关注Visual的多头注意力
        self.text_to_visual_attn = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # 门控机制(防止信息丢失)
        self.gate_v = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Sigmoid()
        )
        
        # Layer Normalization
        self.norm_v = nn.LayerNorm(visual_dim)
        self.norm_t = nn.LayerNorm(text_dim)

        # Feedforward网络(可选,增强表达能力)
        self.ffn_v = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim * 4, visual_dim),
            nn.Dropout(dropout)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 4, text_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_prompt, text_prompt):
        """
        Args:
            visual_prompt: [B, num_tokens, dim] - Visual Prompt特征
            text_prompt: [B, num_tokens, dim] - Text Prompt特征
            
        Returns:
            visual_enhanced: [B, num_tokens, dim] - 增强后的Visual Prompt
            text_enhanced: [B, num_tokens, dim] - 增强后的Text Prompt
        """
        # ========== Step 0: 投影到统一维度 ==========
        v_proj = self.visual_proj(visual_prompt)  # [B, num_v, unified_dim]
        t_proj = self.text_proj(text_prompt)      # [B, num_t, unified_dim]
        
        # ========== Step 1: 双向注意力交互 ==========
        # 转换为(seq_len, batch, dim)格式
        v = v_proj.transpose(0, 1)  # [num_v_tokens, B, unified_dim]
        t = t_proj.transpose(0, 1)    # [num_t_tokens, B, unified_dim]

        # Visual关注Text (Query=Visual, Key=Value=Text)
        v_attended, _ = self.visual_to_text_attn(
            query=v,
            key=t,
            value=t
        )  # [num_v_tokens, B, dim]
        
        # Text关注Visual (Query=Text, Key=Value=Visual)
        t_attended, _ = self.text_to_visual_attn(
            query=t,
            key=v,
            value=v
        )  # [num_t_tokens, B, dim]
        
        # 转回(B, seq_len, dim)
        v_attended = v_attended.transpose(0, 1)  # [B, num_v_tokens, dim]
        t_attended = t_attended.transpose(0, 1)  # [B, num_t_tokens, dim]
        
        v_attended = self.visual_out_proj(v_attended)  # [B, num_v, visual_dim]
        t_attended = self.text_out_proj(t_attended)    # [B, num_t, text_dim]

        # ========== Step 2: 门控融合(保留原始信息) ==========
        gate_v = self.gate_v(visual_prompt)  # [B, num_v_tokens, dim]
        gate_t = self.gate_t(text_prompt)    # [B, num_t_tokens, dim]
        
        # 加权融合: α * 原始特征 + (1-α) * 交互特征
        visual_fused = gate_v * visual_prompt + (1 - gate_v) * v_attended
        text_fused = gate_t * text_prompt + (1 - gate_t) * t_attended
        
        # ========== Step 3: 残差连接 + Layer Norm ==========
        visual_fused = self.norm_v(visual_prompt + visual_fused)
        text_fused = self.norm_t(text_prompt + text_fused)
        
        # ========== Step 4: FFN增强(可选) ==========
        visual_enhanced = visual_fused + self.ffn_v(visual_fused)
        text_enhanced = text_fused + self.ffn_t(text_fused)
        
        return visual_enhanced, text_enhanced


class LightweightCrossModalFusion(nn.Module):
    """
    轻量级跨模态融合模块(如果显存不足可使用此版本)
    使用简化的点积注意力代替多头注意力
    """
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # 简化的门控机制
        self.gate_v = nn.Linear(dim, 1)
        self.gate_t = nn.Linear(dim, 1)
        
    def forward(self, visual_prompt, text_prompt):
        """
        Args:
            visual_prompt: [B, num_v_tokens, dim]
            text_prompt: [B, num_t_tokens, dim]
        """
        # 计算注意力权重
        attn_v2t = torch.bmm(visual_prompt, text_prompt.transpose(1, 2)) * self.scale  # [B, num_v, num_t]
        attn_t2v = torch.bmm(text_prompt, visual_prompt.transpose(1, 2)) * self.scale  # [B, num_t, num_v]
        
        attn_v2t = F.softmax(attn_v2t, dim=-1)
        attn_t2v = F.softmax(attn_t2v, dim=-1)
        
        attn_v2t = self.dropout(attn_v2t)
        attn_t2v = self.dropout(attn_t2v)
        
        # 加权聚合
        v_attended = torch.bmm(attn_v2t, text_prompt)   # [B, num_v, dim]
        t_attended = torch.bmm(attn_t2v, visual_prompt) # [B, num_t, dim]
        
        # 自适应门控
        gate_v = torch.sigmoid(self.gate_v(visual_prompt))  # [B, num_v, 1]
        gate_t = torch.sigmoid(self.gate_t(text_prompt))    # [B, num_t, 1]
        
        visual_enhanced = gate_v * visual_prompt + (1 - gate_v) * v_attended
        text_enhanced = gate_t * text_prompt + (1 - gate_t) * t_attended
        
        return visual_enhanced, text_enhanced
