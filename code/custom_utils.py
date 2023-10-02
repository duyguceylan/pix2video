import itertools
import math
import os
import random 
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from mydiffusers.models.attention import CrossAttention


def create_custom_diffusion(unet, finetune_layers):
    for name, params in unet.named_parameters():
        found = False
        for j in range(len(finetune_layers)):
            if finetune_layers[j] in name:
                params.requires_grad = True
                found = True
                print(name)
                break
        if not found:
            params.requires_grad = False

    #duygu: this is the custom forward function we define for the attention layer
    #duygu: ignore: warp_grid, key_matrix, value_matrix, query_matrix, f_emb
    def new_forward(self, hidden_states, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        selfattn = False
        context = None
        self_attn_query_features_cond=None
        warp_grid=None
        if 'context' in kwargs:
            context = kwargs['context']
        elif 'encoder_hidden_states' in kwargs:
            context = kwargs['encoder_hidden_states']
            selfattn = True

        if 'self_attn_query_features_cond' in kwargs:
            self_attn_query_features_cond=kwargs['self_attn_query_features_cond']
            if self_attn_query_features_cond is not None:
                selfattn = True

        if context is not None:
            crossattn = True
        else:
            #duygu:if self_attn_query_features_cond is not provided we fill it with hidden_states
            #this is equivalent to running the original self attention
            if self_attn_query_features_cond is None and hidden_states.shape[0] > 1:
                self_attn_query_features_cond = torch.zeros_like(hidden_states).to(hidden_states.device)
                j = 0
                while j < hidden_states.shape[0]:
                    self_attn_query_features_cond[j] = hidden_states[j+1]
                    self_attn_query_features_cond[j+1] = hidden_states[j+1]
                    j += 2
                selfattn = True
            elif self_attn_query_features_cond is None:
                #self_attn_query_features_cond = torch.cat([torch.zeros_like(hidden_states).to(hidden_states.device)]*2, dim = 1)
                #self_attn_query_features_cond[:, 0:hidden_states.shape[1], :] = hidden_states
                #self_attn_query_features_cond[:, hidden_states.shape[1]:, :] = hidden_states

                self_attn_query_features_cond = hidden_states
                selfattn = True
            
        context = context if context is not None else hidden_states

        if selfattn:
            query = self.to_q(hidden_states)

            key = self.to_k(self_attn_query_features_cond)
            
            value = self.to_v(self_attn_query_features_cond)
            
            #print('self attn')
            #key = self.to_k(self_attn_query_features_cond)
            #value = self.to_v(self_attn_query_features_cond)
        else:
            query = self.to_q(hidden_states)

            key = self.to_k(context)
            value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def change_forward(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)

    change_forward(unet)
    return unet


def save_progress(unet, save_path, finetune_layers):
    delta_dict = {'unet': {} }
    
    for name, params in unet.named_parameters():
        for j in range(len(finetune_layers)):
            if finetune_layers[j] in name:
                delta_dict['unet'][name] = params.cpu().clone()

    torch.save(delta_dict, save_path)


def load_model(unet, save_path, finetune_layers):
    st = torch.load(save_path)
    
    print(st.keys())
    for name, params in unet.named_parameters():
        for j in range(len(finetune_layers)):
            if finetune_layers[j] in name:
                params.data.copy_(st['unet'][f'{name}'])


def freeze_params(params):
    for param in params:
        param.requires_grad = False