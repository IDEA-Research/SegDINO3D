# SegDINO3D
# url: https://github.com/IDEA-Research/SegDINO3D
# Copyright (c) 2025 IDEA. All Rights Reserved.

import torch
import torch.nn as nn
from mmengine.model import BaseModule
import torch
from torch.utils.checkpoint import checkpoint

from segdino3d import DECODERS
from segdino3d.models.module.utils import PositionEmbeddingCoordsSine, MLP
from segdino3d.models.module.attention import MultiheadAttention
import copy


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs

    def forward_kvq(self, sources_k, sources_v, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources_k)):
            k, v = sources_k[i], sources_v[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, add_positional_embedding=False, 
                 use_activation_checkpoint=False, **kwargs):
        super().__init__()

        # Control whether to return heavy intermediate tensors (hidden states / aux outputs)
        # This can be disabled during training to save GPU memory. Set to True by default.
        self.return_hidden_states = True
        self.return_aux_outputs = True

        # Activation checkpointing to save GPU memory during training (trades compute for memory)
        self.use_activation_checkpoint = use_activation_checkpoint

        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            if add_positional_embedding:
                self.cross_attn_layers.append(MultiheadAttention(d_model*2, num_heads, dropout=dropout, vdim=d_model))
            else:
                self.cross_attn_layers.append(
                    CrossAttentionLayer(
                        d_model, num_heads, dropout, fix_attention))
            if add_positional_embedding:
                self.self_attn_layers.append(MultiheadAttention(d_model, num_heads, dropout=dropout, vdim=d_model))
            else:
                self.self_attn_layers.append(
                    SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_heads = num_heads
    
    def _checkpoint_forward(self, module, *args, **kwargs):
        """Wrapper to apply activation checkpointing on a module's forward pass.
        
        Args:
            module: The module to apply checkpointing to.
            *args, **kwargs: Arguments to pass to module's forward method.
            
        Returns:
            Output of the module's forward pass.
        """
        if self.use_activation_checkpoint and self.training:
            # Use checkpointing only during training to save memory
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        else:
            return module(*args, **kwargs)
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self._checkpoint_forward(self.cross_attn_layers[i], inst_feats, queries)
            queries = self._checkpoint_forward(self.self_attn_layers[i], queries)
            queries = self._checkpoint_forward(self.ffn_layers[i], queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self._checkpoint_forward(self.cross_attn_layers[i], inst_feats, queries, attn_mask)
            queries = self._checkpoint_forward(self.self_attn_layers[i], queries)
            queries = self._checkpoint_forward(self.ffn_layers[i], queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self,
                x, sp_pos=None, sp_pos_wo_elastic=None, queries=None, 
                queries_pos=None, dinox_queries=None, dinox_query_pos=None, scene_range=None,
        ):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries, dinox_queries, sp_pos, queries_pos, dinox_query_pos, sp_pos_wo_elastic, scene_range)
        else:
            return self.forward_simple(x, queries, dinox_queries)

@DECODERS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes, d_model, num_semantic_linears, add_dinox_query_ca=False, 
                 add_dinox_query_ca_mask=False, dinox_query_ca_mask_threshold=0.2, mask_attention_threshold=0.5, 
                 add_positional_embedding=False, pos_type="fourier", temperature=10000, gauss_scale=1.0,
                 add_box_size_pred=False, box_modulate_ca=False, normalize_box_prediction=False, 
                 **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, add_positional_embedding=add_positional_embedding, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)
        
        # if add additional cross-attention to dinox's output queries. 
        self.add_dinox_query_ca = add_dinox_query_ca
        if self.add_dinox_query_ca:
            self.dinox_query_cross_attn_layers = nn.ModuleList([])
            for i in range(kwargs["num_layers"]):
                self.dinox_query_cross_attn_layers.append(
                    CrossAttentionLayer(
                        d_model, kwargs["num_heads"], kwargs["dropout"], kwargs["fix_attention"]))
        
        self.add_dinox_query_ca_mask = add_dinox_query_ca_mask
        self.dinox_query_ca_mask_threshold = dinox_query_ca_mask_threshold
        self.mask_attention_threshold = mask_attention_threshold
        self.num_semantic_classes = num_semantic_classes
        self.add_positional_embedding = add_positional_embedding
        self.add_box_size_pred = add_box_size_pred
        if self.add_positional_embedding:
            # for positional embedding
            self.position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=True, pos_type=pos_type, d_pos=d_model, gauss_scale=gauss_scale)
            self.ref_point_head = MLP(d_model, d_model, d_model, 2)    # use to project pos embedding
            # for center regression
            _bbox_embed = MLP(d_model, d_model, 3, 3)
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(kwargs["num_layers"])
            ]
            self.bbox_embed = nn.ModuleList(box_embed_layerlist)    # update positional query, predict a offset
            # for projection in cross-attention
            self.ca_qcontent_proj = nn.ModuleList([])
            self.ca_qpos_proj = nn.Linear(d_model, d_model)
            self.ca_kcontent_proj = nn.ModuleList([])
            self.ca_kpos_proj = nn.ModuleList([])
            self.ca_v_proj = nn.ModuleList([])
            self.ca_qpos_sine_proj = nn.ModuleList([])
            self.norm1 = nn.ModuleList([])
            self.dropout1 = nn.ModuleList([])
            for i in range(kwargs["num_layers"]):
                self.ca_qcontent_proj.append(nn.Linear(d_model, d_model))
                self.ca_kcontent_proj.append(nn.Linear(d_model, d_model))
                self.ca_kpos_proj.append(nn.Linear(d_model, d_model))
                self.ca_v_proj.append(nn.Linear(d_model, d_model))
                self.ca_qpos_sine_proj.append(nn.Linear(d_model, d_model))
                self.norm1.append(nn.LayerNorm(d_model))
                self.dropout1.append(nn.Dropout(kwargs["dropout"]))
            # for projection in self-attention
            self.sa_qcontent_proj = nn.ModuleList([])
            self.sa_qpos_proj = nn.ModuleList([])
            self.sa_kcontent_proj = nn.ModuleList([])
            self.sa_kpos_proj = nn.ModuleList([])
            self.sa_v_proj = nn.ModuleList([])
            self.norm2 = nn.ModuleList([])
            self.dropout2 = nn.ModuleList([])
            for i in range(kwargs["num_layers"]):
                self.sa_qcontent_proj.append(nn.Linear(d_model, d_model))
                self.sa_qpos_proj.append(nn.Linear(d_model, d_model))
                self.sa_kcontent_proj.append(nn.Linear(d_model, d_model))
                self.sa_kpos_proj.append(nn.Linear(d_model, d_model))
                self.sa_v_proj.append(nn.Linear(d_model, d_model))
                self.norm2.append(nn.LayerNorm(d_model))
                self.dropout2.append(nn.Dropout(kwargs["dropout"]))
            # for center regression
            if self.add_box_size_pred:
                _bbox_size_embed = MLP(d_model, d_model, 3, 3)
                nn.init.constant_(_bbox_size_embed.layers[-1].weight.data, 0)
                nn.init.constant_(_bbox_size_embed.layers[-1].bias.data, 0)
                box_size_embed_layerlist = [
                    copy.deepcopy(_bbox_size_embed) for i in range(kwargs["num_layers"])
                ]
                self.bbox_size_embed = nn.ModuleList(box_size_embed_layerlist)    # update positional query, predict a offset
        self.box_modulate_ca = box_modulate_ca
        if self.box_modulate_ca:
            assert self.add_positional_embedding and self.add_box_size_pred, " If you want to use box to modulate cross attention, you should set add_positional_embedding and add_box_size_pred to True."
            assert pos_type=="sine", "Only implemented for sine positional embedding now."
            self.ref_anchor_head = MLP(d_model, d_model, 3, 2)
        self.normalize_box_prediction = normalize_box_prediction

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            feats_for_sem = norm_query
            feats_for_inst = norm_query
            cls_preds.append(self.out_cls(feats_for_inst))
            if last_flag:
                sem_preds.append(self.out_sem(feats_for_sem))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < self.mask_attention_threshold).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, 
            x, queries, dinox_queries = None,
            x_pos = None, queries_pos = None,
            dinox_query_pos = None,
            pos_wo_elastic = None,
            scene_range = None,
        ):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_queries_i, in_channles).
            x_pos (List[Tensor]): of len batch_size, each of shape
                (n_points_i, 3).
            queries_pos (List[Tensor], optional): of len batch_size, each of shape
                (n_queries_i, 3).
            scene_range : (List[Tuple(min_coords, max_coords)], optional): of len batch_size.
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        if self.add_positional_embedding: 
            assert (x_pos is not None) and (queries_pos is not None) and (scene_range is not None)
            positional_queries = queries_pos # List[Tensor] of len batch_size, each of shape (n_queries, 3), metric coordinates
            reference_points = positional_queries
            memory_emb = [self.position_embedding(pos.unsqueeze(0), input_range=(scene_range[j][0].unsqueeze(0), scene_range[j][1].unsqueeze(0)))[0] for j, pos in enumerate(x_pos)]    # (n_points, d_model)
            if self.normalize_box_prediction:
                bbox_size_queries = [1/(s_range[1] - s_range[0])*0.5 for s_range in scene_range]
            else:
                bbox_size_queries = [torch.ones_like(query_pos)*0.5 for query_pos in queries_pos]  # (n_queries, 3)
            reference_sizes = bbox_size_queries
        else:
            positional_queries = None
            bbox_size_queries = None
        cls_preds, sem_preds, pred_scores, pred_masks, pred_centers, pred_sizes = [], [], [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        if self.add_positional_embedding:
            pred_centers.append([None for _ in range(len(queries))])
            pred_sizes.append([None for _ in range(len(queries))])
        for i in range(len(self.cross_attn_layers)):
            # ========== begin cross-attention ==========
            if self.add_positional_embedding:
                if self.box_modulate_ca:
                    refHWL_cond = [self.ref_anchor_head(query).sigmoid() for query in queries] # (n_queries, 3)
                    modulated_coe = [refHWL_cond[j] / reference_sizes[j] for j in range(len(refHWL_cond))]  # (n_queries, 3)
                    positional_queries_emb = [self.position_embedding(pos.unsqueeze(0), input_range=(scene_range[j][0].unsqueeze(0), scene_range[j][1].unsqueeze(0)), \
                                                modulated=modulated_coe[j].unsqueeze(0))[0] for j, pos in enumerate(reference_points)]   # (n_queries, d_model)
                else:
                    positional_queries_emb = [self.position_embedding(pos.unsqueeze(0), input_range=(scene_range[j][0].unsqueeze(0), scene_range[j][1].unsqueeze(0)))[0] for j, pos in enumerate(reference_points)]   # (n_queries, d_model)
                query_pos = [self.ref_point_head(y) for y in positional_queries_emb]    # (n_queries, d_model), MLP(PE)

                q_content = [self.ca_qcontent_proj[i](query) for query in queries]  # (n_queries, d_model)
                k_content = [self.ca_kcontent_proj[i](inst_feat) for inst_feat in inst_feats]   # (n_points, d_model)
                v = [self.ca_v_proj[i](inst_feat) for inst_feat in inst_feats]  # (n_points, d_model)
                k_pos = [self.ca_kpos_proj[i](memory_emb[j]) for j in range(len(memory_emb))]   # (n_points, d_model)
                if i == 0:
                    q_pos = [self.ca_qpos_proj(pos) for pos in query_pos]   # (n_queries, d_model)
                    q = [q_content[j] + q_pos[j] for j in range(len(q_content))]
                    k = [k_content[j] + k_pos[j] for j in range(len(k_content))]
                else:
                    q = q_content
                    k = k_content
                for j in range(len(queries)):
                    num_queries = queries[j].shape[0]
                    q_j = q[j].view(num_queries, 1, self.num_heads, self.d_model // self.num_heads)
                    positional_queries_emb_j = self.ca_qpos_sine_proj[i](positional_queries_emb[j]) # (n_queries, d_model)
                    q_pos_j = positional_queries_emb_j.view(num_queries, 1, self.num_heads, self.d_model // self.num_heads)
                    q_j = torch.cat([q_j, q_pos_j], dim=3).view(num_queries, 1, self.d_model * 2)   # (n_queries, 1, d_model * 2)
                    k_j = k[j].view(-1, 1, self.num_heads, self.d_model // self.num_heads)
                    k_pos_j = k_pos[j].view(-1, 1, self.num_heads, self.d_model // self.num_heads)
                    k_j = torch.cat([k_j, k_pos_j], dim=3).view(-1, 1, self.d_model * 2)    # (n_points, 1, d_model * 2)
                    _fn = lambda q, k, v, m, module=self.cross_attn_layers[i]: module(query=q, key=k, value=v, attn_mask=m)[0].squeeze(1)
                    tgt2 = self._checkpoint_forward(_fn, q_j, k_j, v[j].unsqueeze(1), attn_mask[j].unsqueeze(0).expand(self.num_heads, -1, -1))  # (n_queries, d_model)
                    queries[j] = queries[j] + self.dropout1[i](tgt2)
                    queries[j] = self.norm1[i](queries[j])
            else:
                queries = self._checkpoint_forward(self.cross_attn_layers[i], inst_feats, queries, attn_mask)
            # ========== begin self-attention ==========
            if self.add_positional_embedding:
                q_content = [self.sa_qcontent_proj[i](query) for query in queries]  # (n_queries, d_model)
                q_pos = [self.sa_qpos_proj[i](query_pos_j) for query_pos_j in query_pos]    # (n_queries, d_model)
                k_content = [self.sa_kcontent_proj[i](query) for query in queries]  # (n_queries, d_model)
                k_pos = [self.sa_kpos_proj[i](query_pos_j) for query_pos_j in query_pos]    # (n_queries, d_model)
                v = [self.sa_v_proj[i](query) for query in queries] # (n_queries, d_model)
                for j in range(len(queries)):
                    q_j = q_content[j] + q_pos[j]  # (n_queries, d_model)
                    k_j = k_content[j] + k_pos[j]  # (n_queries, d_model)
                    q_j = q_j.view(-1, 1, self.d_model) # (n_queries, 1, d_model)
                    k_j = k_j.view(-1, 1, self.d_model) # (n_queries, 1, d_model)
                    _fn = lambda q, k, v, module=self.self_attn_layers[i]: module(q, k, value=v, attn_mask=None)[0].squeeze(1)
                    tgt2 = self._checkpoint_forward(_fn, q_j, k_j, v[j].unsqueeze(1))  # (nq, d_model)
                    queries[j] = queries[j] + self.dropout2[i](tgt2)
                    queries[j] = self.norm2[i](queries[j])
            else:
                queries = self._checkpoint_forward(self.self_attn_layers[i], queries)
            # ========== begin 2d query cross-attention ==========
            if self.add_dinox_query_ca: 
                if self.add_dinox_query_ca_mask:
                    # attn_mask: num_q X num_sp
                    dinox_query_ca_attn_mask = []
                    dinox_queries_ = []
                    for batch_id in range(len(pos_wo_elastic)):
                        if type(dinox_query_pos[batch_id]) != torch.Tensor:
                            dinox_query_pos[batch_id] = dinox_query_pos[batch_id].tensor.type(pos_wo_elastic[batch_id].dtype).to(pos_wo_elastic[batch_id].device)
                        dist = torch.cdist(pos_wo_elastic[batch_id], dinox_query_pos[batch_id], p=1)  # num_sp X num_dinoxq
                        mask_ = (~attn_mask[batch_id]).float() @ (dist < self.dinox_query_ca_mask_threshold).float()
                        mask_ = mask_ == 0
                        # to prevent the situation where no dinox query is matched with a 3D query.
                        dinox_queries_.append(torch.cat([dinox_queries[batch_id], dinox_queries[batch_id].new_ones(1, dinox_queries[batch_id].shape[1])], dim=0))
                        mask_ = torch.cat([mask_, mask_.new_zeros(mask_.shape[0], 1)], dim=-1)
                        dinox_query_ca_attn_mask.append(mask_)

                    queries = self._checkpoint_forward(self.dinox_query_cross_attn_layers[i], dinox_queries_, queries, dinox_query_ca_attn_mask)
                else:
                    queries = self._checkpoint_forward(self.dinox_query_cross_attn_layers[i], dinox_queries, queries)
            # ========== begin ffn ==========
            queries = self._checkpoint_forward(self.ffn_layers[i], queries)
            # ========== update positional query ==========
            if self.add_positional_embedding:
                pred_center = []
                for j in range(len(queries)):
                    pred_center.append(positional_queries[j] + self.bbox_embed[i](queries[j]))
                pred_centers.append(pred_center)
                reference_points = [center.detach() for center in pred_center]
                positional_queries = reference_points
                if self.add_box_size_pred:
                    pred_size = []
                    for j in range(len(queries)):
                        if self.normalize_box_prediction:
                            pred_size.append((
                                inverse_sigmoid(bbox_size_queries[j]) + \
                                self.bbox_size_embed[i](queries[j])
                            ).sigmoid())
                        else:
                            pred_size.append(bbox_size_queries[j] + self.bbox_size_embed[i](queries[j]))
                    pred_sizes.append(pred_size)
                    reference_sizes = [size.detach() for size in pred_size]
                    bbox_size_queries = reference_sizes
                else:
                    pred_sizes.append([None for _ in range(len(queries))])
            else:
                pred_centers.append([None for _ in range(len(queries))])
                pred_sizes.append([None for _ in range(len(queries))])
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        if self.normalize_box_prediction:
            for l_id in range(len(pred_sizes)):
                if pred_sizes[l_id][0] is not None:
                    for b_id in range(len(pred_sizes[l_id])):
                        pred_sizes[l_id][b_id] = pred_sizes[l_id][b_id] * (scene_range[b_id][1] - scene_range[b_id][0])
        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores,
                centers=centers,
                sizes=sizes)
            for cls_pred, sem_pred, scores, masks, centers, sizes in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1], pred_centers[:-1], pred_sizes[:-1])]

        # Build result dict. Optionally skip heavy tensors to save GPU memory.
        result = dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            centers=pred_centers[-1],
            sizes=pred_sizes[-1])

        if getattr(self, 'return_hidden_states', True):
            result['hidden_states'] = queries
        if getattr(self, 'return_aux_outputs', True):
            result['aux_outputs'] = aux_outputs

        return result
