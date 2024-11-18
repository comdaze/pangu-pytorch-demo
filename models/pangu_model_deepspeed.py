import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from collections import OrderedDict

import torch
from torch import nn

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

from models.layers import *
from era5_data import utils_data

class PanguModelPipe(PipelineModule):
    def __init__(self, num_stages=4, depths=[2, 6, 6, 2], num_heads=[6, 12, 12, 6], 
                 dims=[192, 384, 384, 192], patch_size=(2, 4, 4), device=None):
        self.device = device
        
        # Define pipeline stages
        self.layers = [
            LayerSpec(nn.Sequential, OrderedDict([
                ('embed', PatchEmbedding_pretrain(patch_size, dims[0])),
                ('layer0', EarthSpecificLayer(
                    depth=depths[0],
                    dim=dims[0],
                    drop_path_ratio_list=[x.item() for x in torch.linspace(0, 0.2, depths[0])],
                    heads=num_heads[0],
                    use_checkpoint=True,
                    device=device
                ))
            ])),
            LayerSpec(nn.Sequential, OrderedDict([
                ('downsample', DownSample(dims[0])),
                ('layer1', EarthSpecificLayer(
                    depth=depths[1],
                    dim=dims[1],
                    drop_path_ratio_list=[x.item() for x in torch.linspace(0, 0.2, depths[1])],
                    heads=num_heads[1],
                    use_checkpoint=True,
                    device=device
                ))
            ])),
            LayerSpec(EarthSpecificLayer,
                      depth=depths[2],
                      dim=dims[2],
                      drop_path_ratio_list=[x.item() for x in torch.linspace(0, 0.2, depths[2])],
                      heads=num_heads[2],
                      use_checkpoint=True,
                      device=device),
            LayerSpec(nn.Sequential, OrderedDict([
                ('upsample', UpSample(dims[-2], dims[-1])),
                ('layer3', EarthSpecificLayer(
                    depth=depths[3],
                    dim=dims[3],
                    drop_path_ratio_list=[x.item() for x in torch.linspace(0, 0.2, depths[3])],
                    heads=num_heads[3],
                    use_checkpoint=True,
                    device=device
                )),
                ('output', PatchRecovery_pretrain(dims[-2]))
            ])),
        ]
        
        super().__init__(
            layers=self.layers,
            num_stages=num_stages,
            loss_fn=None
        )
        
        # Save dimension info for forward
        self.dims = dims
        self.spatial_size = (8, 181, 360)
        self.reduced_size = (8, 91, 180)

    def forward(self, inputs):
        if self.stage_id == 0:
            input, input_surface, statistics, maps, const_h = inputs[0]
            x = self.layers[0](input, input_surface, statistics, maps, const_h)
            return x
            
        elif self.stage_id == 1:
            x = inputs
            x = self.layers[1](x)
            return x
            
        elif self.stage_id == 2:
            x = inputs
            x = self.layers[2](x)
            return x
            
        else:
            x, skip = inputs
            x = self.layers[3](x)
            x = torch.cat((skip, x), dim=-1)
            output, output_surface = x
            return output, output_surface

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    deepspeed.init_distributed(dist_backend='nccl')

    model = PanguModelPipe(device=device).to(device)
    
    # Initialize DeepSpeed
    ds_config = "ds_config.json"
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    # Load all statistics and constants
    aux_constants = utils_data.loadAllConstants(device=device)

    x_surface = torch.randn((1, 4, 721, 1440)).to(device)
    x_upper = torch.randn((1, 5, 13, 721, 1440)).to(device)
    data_iter = iter([(x_upper, x_surface, aux_constants['weather_statistics'], aux_constants['constant_maps'], aux_constants['const_h'])])
    
    output, output_surface = model_engine.train_batch(data_iter=data_iter)
    print(output.shape)