import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from collections import OrderedDict

import torch
from torch import nn

from models.layers import *
from era5_data import utils_data

# import torch.utils.checkpoint as checkpoint


class PanguModel(nn.Module):
    def __init__(self, depths=[2, 6, 6, 2], num_heads=[6, 12, 12, 6], dims=[192, 384, 384, 192], patch_size=(2, 4, 4), device=None):
        super(PanguModel, self).__init__()

        # Patch embedding
        self.device = device

        # self._input_layer = PatchEmbedding(patch_size, dims[0], device=self.device)
        self._input_layer = PatchEmbedding_pretrain(patch_size, dims[0])
        self.downsample = DownSample(dims[0])

        dpr = [x.item() for x in torch.linspace(0, 0.2, sum(depths))]
        # build layers
        self.num_layers = len(depths)

        layer_list = OrderedDict()
        for i_layer in range(self.num_layers):
            layer_list['EarthSpecificLayer{}'.format(i_layer)] = EarthSpecificLayer(
                depth=depths[i_layer],
                dim=dims[i_layer],
                drop_path_ratio_list=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                heads=num_heads[i_layer],
                use_checkpoint=self.training,
                device=self.device)
        self.layers = nn.Sequential(layer_list)

        self.upsample = UpSample(dims[-2], dims[-1])

        # Patch Recovery
        # self._output_layer = PatchRecovery(dims[-2])
        self._output_layer = PatchRecovery_pretrain(dims[-2])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input, input_surface, statistics, maps, const_h):
        '''Backbone architecture'''
        # Embed the input fields into patches
        # input:(B, N, Z, H, W) ([1, 5, 13, 721, 1440]) input_surface:(B, N, H, W)([1, 4, 721, 1440])
        # x = checkpoint.checkpoint(self._input_layer, input, input_surface)
        # ([1, 521280, 192]) [B, spatial, C]
        
        # print(f"Input shape: {input.shape}")
        # print(f"Input surface shape: {input_surface.shape}")
        
        x = self._input_layer(input, input_surface, statistics, maps, const_h)
        # print(f"After input layer shape: {x.shape}")

        # Encoder, composed of two layers
        # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper

        x = self.layers[0](x, 8, 181, 360)
        # print(f"After first layer shape: {x.shape}")

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (8, 360, 181) to (8, 180, 91)
        x = self.downsample(x, 8, 181, 360)

        x = self.layers[1](x, 8, 91, 180)
        # Decoder, composed of two layers
        # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
        x = self.layers[2](x, 8, 91, 180)

        # Upsample from (8, 180, 91) to (8, 360, 181)
        x = self.upsample(x)

        # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
        x = self.layers[3](x, 8, 181, 360)  # ([1, 521280, 192])

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat((skip, x), dim=-1)

        # Recover the output fields from patches
        # output, output_surface = checkpoint.checkpoint(self._output_layer, x, 8, 181, 360)
        output, output_surface = self._output_layer(x, 8, 181, 360)

        return output, output_surface


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PanguModel(device=device).to(device)
    
    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device)

    x_surface = torch.randn((1, 4, 721, 1440)).to(device)
    x_upper = torch.randn((1, 5, 13, 721, 1440)).to(device)
    output, output_surface = model(x_upper, x_surface, aux_constants['weather_statistics'], aux_constants['constant_maps'], aux_constants['const_h'])
    # print(output)
    print(output.shape)
    
    # batch_size=2
    x_surface = torch.randn((2, 4, 721, 1440)).to(device)
    x_upper = torch.randn((2, 5, 13, 721, 1440)).to(device)
    output, output_surface = model(x_upper, x_surface, aux_constants['weather_statistics'], aux_constants['constant_maps'], aux_constants['const_h'])
    # print(output)
    print(output.shape)
