__all__ = ['PatchTST_backbone']
# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.DishTS import DishTS
from layers.SPM_layers import *

# Cell
class SPM_backbone(nn.Module):
    def __init__(self, configs, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        super().__init__()
        # RevIn
        self.revin = revin
        self.dishts = configs.dishts
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        if self.dishts: self.DishTS = DishTS(configs)

        # Backbone 
        self.backbone = TSTiEncoder(configs, target_window=target_window, context_window=context_window, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.predict = MLP(configs.num_layers, (d_model-1)*2, [x*96 for x in configs.hidden_size], target_window)
        
    def forward(self, z):                                                                   # [b, c, l]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm') 
            if self.dishts: 
                z, _ = self.DishTS(z, 'forward', None)  
            z = z.permute(0,2,1)
            
        B, C, L = z.shape
        z = torch.fft.rfft2(z, dim=(1, 2), norm='ortho')                                    # [b, c, l/2]                             
        z = self.backbone(z)                                                                # [b, c, d]   
        z = torch.fft.irfft2(z, s=(C, (z.size(-1) - 1)*2), dim=(1, 2), norm="ortho")        # [b, c, (d-1)*2]  
        z = self.predict(z)
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            if self.dishts: 
                z = self.DishTS(z, 'inverse', None)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
      

class MLP(nn.Module):
    def __init__(self, num_layers, in_features, hidden_sizes, out_features):
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module(f'linear_{0}', nn.Linear(in_features, hidden_sizes[0]))
        self.mlp.add_module(f'leaky_relu_{0}', nn.LeakyReLU())  
        for i in range(1, num_layers - 1):
            self.mlp.add_module(f'linear_{i}', nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.mlp.add_module(f'leaky_relu_{i}', nn.LeakyReLU())
        self.mlp.add_module(f'linear_{num_layers - 1}', nn.Linear(hidden_sizes[-1], out_features))
        
    def forward(self, x):  
        x = self.mlp(x)
        return x
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, configs, target_window, context_window, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()
        self.encoder = TSTEncoder(configs, target_window, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        self.embeddings = nn.Linear(context_window//2+1, d_model).to(torch.cfloat)
                                                              
    def forward(self, x) -> Tensor:                                              # [b, c, L/2+1]
        b,c,l = x.size()        
        x = self.embeddings(x)                                                   # [b, c, d]
        o1_real = x.real                                                         # [b, c, d]
        o1_imag = x.imag                                                         # [b, c, d]
        y = torch.cat([o1_real, o1_imag], dim=0)                                 # [2*b, c, d]   
        y = self.encoder(y)                                                      # [2*b, c, d]
        y = torch.stack([y[:b], y[b:]], dim=-1)                                  # [b, c, d, 2] 
        y = torch.view_as_complex(y.contiguous())                                # [b, c, d]    
        return y  
   
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, configs, target_window, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(configs, target_window, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, configs, target_window, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),    
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)
        
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)
        
        if self.res_attention:
            return src, scores
        else:
            return src



class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # # Linear (+ split in multiple heads)
        # q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        # k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # k_s    : [bs x n_heads x max_q_len x d_k]
        # v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        
        # Linear (+ split in multiple heads)
        q_s = Q # self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = torch.flip(K, dims=[1, 2]) # self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # k_s    : [bs x n_heads x max_q_len x d_k]
        v_s = V # self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]
        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        # output = self.to_out(output)
        
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        # CONSTRUCT COMPLEX NUMBER METRICS, q_re: torch.Size([b, 1, c, d]) k_re: torch.Size([b, 1, c, d]) v_re: torch.Size([b, 1, c, d])
        q_re, q_im = q[:q.size(0)//2], q[q.size(0)//2:]
        k_re, k_im = k[:k.size(0)//2], k[k.size(0)//2:]
        v_re, v_im = v[:v.size(0)//2], v[:v.size(0)//2]
        # print('q:', q_re.shape, q_im.shape)
        
        # compute scores, attn_scores: torch.Size([b, 1, c, d])
        attn_scores_real = self.scale * (torch.mul(q_re, k_re)-torch.mul(q_im, k_im))        # Element-wise multiplication
        attn_scores_imag = self.scale * (torch.mul(q_re, k_im)-torch.mul(q_im, k_re)) 
        attn_scores = torch.cat([attn_scores_real, attn_scores_imag], dim=0)     
        # print('attn_scores:', attn_scores.shape)        
        
        # compute attention output
        output_real = torch.mul(attn_scores_real, v_re) - torch.mul(attn_scores_imag, v_im)                       
        output_imag = torch.mul(attn_scores_real, v_im) + torch.mul(attn_scores_imag, v_re)                       
        output = torch.cat([output_real, output_imag], dim=0)  
        # print('output:', output.shape)

        if self.res_attention: return output, attn_scores, attn_scores
        else: return output, attn_scores
