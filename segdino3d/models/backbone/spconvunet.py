# ------------------------------------------------------------------------
# SegDINO3D
# url: https://github.com/IDEA-Research/SegDINO3D
# Copyright (c) 2025 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from: SPFormer (https://github.com/sunjiahao1999/SPFormer)
# ------------------------------------------------------------------------

import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn
import MinkowskiEngine as ME
from torch_scatter import scatter_mean
from segdino3d import BACKBONES


class ResidualBlock(SparseModule):
    """Resudual block for SpConv U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int: Number of output channels.
        norm_fn (Callable): Normalization function constructor.
        indice_key (str): SpConv key for conv layer.
        normalize_before (bool): Wheter to call norm before conv.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_fn=functools.partial(
                     nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None,
                 normalize_before=True):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size=1, bias=False))

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key), norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input tensor.
        
        Returns:
            SparseConvTensor: Output tensor.
        """
        identity = spconv.SparseConvTensor(input.features, input.indices,
                                           input.spatial_shape,
                                           input.batch_size)

        output = self.conv_branch(input)
        output = output.replace_feature(output.features +
                                        self.i_branch(identity).features)

        return output


@BACKBONES.register_module()
class SpConvUNet(nn.Module):
    """SpConv U-Net model.

    Args:
        num_planes (List[int]): Number of channels in each level.
        norm_fn (Callable): Normalization function constructor.
        block_reps (int): Times to repeat each block.
        block (Callable): Block base class.
        indice_key_id (int): Id of current level.
        normalize_before (bool): Wheter to call norm before conv. 
        return_blocks (bool): Whether to return previous blocks.
    """

    def __init__(self,
                 num_planes,
                 norm_fn=functools.partial(
                      nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2,
                 block=ResidualBlock,
                 indice_key_id=1,
                 normalize_before=True,
                 return_blocks=False,
                 voxel_size=0.02,
                 mode_fuse_2d_feat="early_fusion",
                 main_model=True,
                 min_spatial_shape=128,
                 add_positional_embedding=False):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        # process block and norm_fn caller
        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock
        blocks = {
            f'block{i}': block(
                num_planes[0],
                num_planes[0],
                norm_fn,
                normalize_before=normalize_before,
                indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(num_planes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]), nn.ReLU(),
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[1]), nn.ReLU())

            self.u = SpConvUNet(
                num_planes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                main_model=False,)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[0]), nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    num_planes[0] * (2 - i),
                    num_planes[0],
                    norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)
        self.mode_fuse_2d_feat = mode_fuse_2d_feat
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape

        self.main_model = main_model
        if self.main_model:
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(
                    256 + 6 if mode_fuse_2d_feat.startswith("early_fusion") else 256,
                    32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key='subm1'))
            self.output_layer = spconv.SparseSequential(
                torch.nn.BatchNorm1d(32, eps=1e-4, momentum=0.1),
                torch.nn.ReLU(inplace=True))
        
        self.add_positional_embedding = add_positional_embedding

    def forward(self, input, previous_outputs=None):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input tensor.
            previous_outputs (List[SparseConvTensor]): Previous imput tensors.

        Returns:
            SparseConvTensor: Output tensor.
        """
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices,
                                           output.spatial_shape,
                                           output.batch_size)

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(
                    output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            # NOTE: to avoid the residual bug
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output

    def collate(self, points, elastic_points=None, points_rgbfeat=None, sp_pts_masks=None, batch_offsets=None, return_sp_mean_pos = True):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            if self.mode_fuse_2d_feat.startswith("early_fusion"):
                coordinates, features = ME.utils.batch_sparse_collate(
                    [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                      torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0), p_f)))
                     for p, p_f in zip(points, points_rgbfeat)])
        else:
            if self.mode_fuse_2d_feat.startswith("early_fusion"):
                coordinates, features = ME.utils.batch_sparse_collate(
                    [((el_p - el_p.min(0)[0]),
                    torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0), p_f)))
                    for el_p, p, p_f in zip(elastic_points, points, points_rgbfeat)])
            else:
                coordinates, features = ME.utils.batch_sparse_collate(
                    [((el_p - el_p.min(0)[0]),
                    torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                    for el_p, p in zip(elastic_points, points)])
        dinox_feat = []
        for i in range(len(points)):
            if points_rgbfeat is not None:
                dinox_feat.append(points_rgbfeat[i])
        if len(dinox_feat) > 0:
            dinox_feat = torch.cat(dinox_feat, dim=0)
        else:
            dinox_feat = None
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        if return_sp_mean_pos:
            coordinates_wo_elastic, features_wo_elastic = [], []
            for i in range(len(points)):
                coordinates_wo_elastic.append(points[i][:, :3])
                features_wo_elastic.append(points[i][:, 3:])
            coordinates_wo_elastic, features_wo_elastic = ME.utils.batch_sparse_collate(
                [(c / self.voxel_size, f) for c, f in zip(coordinates_wo_elastic, features_wo_elastic)],
                device=coordinates_wo_elastic[0].device)
            x_mean_coords_wo_elastic = scatter_mean(
                coordinates_wo_elastic[..., 1:].float() * self.voxel_size, 
                torch.cat(sp_pts_masks)
            , dim=0)  # get the center position of superpoints.
            pos_wo_elastic = []
            for i in range(len(batch_offsets)-1):
                begin = batch_offsets[i]
                end = batch_offsets[i+1]
                pos_wo_elastic.append(x_mean_coords_wo_elastic[begin: end])
        else:
            pos_wo_elastic = None

        if self.add_positional_embedding:
            coordinates_w_elastic, features_w_elastic = [], []
            for i in range(len(points)):
                if elastic_points is not None:
                    coordinates_w_elastic.append(elastic_points[i][:, :3])
                    features_w_elastic.append(elastic_points[i][:, 3:])
                else:
                    coordinates_w_elastic.append(points[i][:, :3] / self.voxel_size)
                    features_w_elastic.append(points[i][:, 3:])
            coordinates_w_elastic, features_w_elastic = ME.utils.batch_sparse_collate(
                [(c, f) for c, f in zip(coordinates_w_elastic, features_w_elastic)],
                device=coordinates_w_elastic[0].device)
            field_ = ME.TensorField(coordinates=coordinates_w_elastic, features=features_w_elastic)
            x_mean_coords_w_elastic = scatter_mean(
                coordinates_w_elastic[..., 1:].float() * self.voxel_size, 
                torch.cat(sp_pts_masks)
            , dim=0)  # get the center position of superpoints.
            pos = []
            for i in range(len(batch_offsets)-1):
                begin = batch_offsets[i]
                end = batch_offsets[i+1]
                pos.append(x_mean_coords_w_elastic[begin: end])
        else:
            pos = None

        return coordinates, features, dinox_feat, inverse_mapping, spatial_shape, pos, pos_wo_elastic

    def forward_wrapper(self, samples, targets, return_sp_mean_pos=True):
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        for i in range(len(samples)):
            gt_pts_seg = targets[i]["extra_features"]["super_point_masks"].clone()  # 
            gt_pts_seg += superpoint_bias
            superpoint_bias = gt_pts_seg.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg)

        coordinates, features, x_dinox, inverse_mapping, spatial_shape, x_pos, pos_wo_elastic = self.collate(
            samples,
            [tgt["elastic_coords"] for tgt in targets] if 'elastic_coords' in targets[0] else None,
            [tgt["extra_features"]["points_2dfeats"] for tgt in targets] if 'points_2dfeats' in targets[0]['extra_features'] else None,
            sp_pts_masks=sp_pts_masks,
            batch_offsets=batch_offsets,
            return_sp_mean_pos=return_sp_mean_pos)

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)

        x = self.input_conv(x)
        x, _ = self.forward(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], sp_pts_masks, dim=0)
        if x_dinox is not None:
            x_dinox = scatter_mean(x_dinox, sp_pts_masks, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])
        if return_sp_mean_pos:
            return out, x_pos, pos_wo_elastic
        else:
            return out, x_pos
        