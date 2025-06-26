from types import SimpleNamespace
import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.models.video.swin_transformer import SwinTransformer3d
from timm.models._manipulate import checkpoint

class SwinEncoder3d(SwinTransformer3d):
    """
    A 3D Swin Transformer encoder that extracts multi-scale features

    """
    def __init__(self, cfg: SimpleNamespace, use_checkpoint: bool = False):
        # Initialize the parent SwinTransformer3d class with its default parameters
        new_patch_size = [32, 4, 4]
        super().__init__(
            patch_size=new_patch_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[8, 7, 7],
            stochastic_depth_prob=0.2,
        )
        self.use_checkpoint = use_checkpoint

        # 1. Load the state dictionary from the original pretrained model
        original_model_weights = swin3d_t(weights=Swin3D_T_Weights.DEFAULT).state_dict()

        # 2. Remove the weight and bias for the layer that has a shape mismatch
        original_patch_embed_weight = original_model_weights.pop("patch_embed.proj.weight")
        original_patch_embed_bias = original_model_weights.pop("patch_embed.proj.bias")
        
        # 3. Load all the other weights that perfectly match
        self.load_state_dict(original_model_weights, strict=False)

        # 4. initialize the new patch_embed layer
        
        # Create the new projection layer with the correct in_channels and kernel_size
        new_proj = nn.Conv3d(
            cfg.in_chans, 96, kernel_size=new_patch_size, stride=new_patch_size
        )

        # Initialize its weights by averaging the original pretrained weights.
        # This transfers knowledge from the original model to our new layer.
        # We average across the old depth dimension (dim=2) and repeat for the new depth.
        new_weight = original_patch_embed_weight.mean(dim=2, keepdim=True).repeat(1, 1, new_patch_size[0], 1, 1)
        
        # Adapt for different input channels if necessary
        if cfg.in_chans != 3:
            new_weight = new_weight.mean(dim=1, keepdim=True).repeat(1, cfg.in_chans, 1, 1, 1)
        
        new_proj.weight.data = new_weight
        new_proj.bias.data = original_patch_embed_bias # The bias shape is unchanged

        # Replace the model's patch_embed projection layer with our new one
        self.patch_embed.proj = new_proj

        # Determine the output channels from each stage for the decoder
        with torch.no_grad():
            dummy_input = torch.randn(1, cfg.in_chans, *cfg.roi_size)
            self.channels = [f.shape[1] for f in self.forward(dummy_input)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Overrides the default forward pass to return a list of feature maps.
        """
        res = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Iterate through the flat list of stages and downsamplers
        for i, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            
            # A 'stage' is an even-indexed module in the features list.
            # After each stage, we save the feature map.
            if i % 2 == 0:
                # The output tensor is [B, D, H, W, C] (channels-last).
                # Permute to [B, C, D, H, W] (channels-first) for standard use.
                res.append(x.permute(0, 4, 1, 2, 3))
            
        return res

if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = SimpleNamespace()
    # This roi_size requires padding, which is handled inside PatchEmbed3d
    cfg.roi_size = (96, 128, 128)
    cfg.in_chans = 1
    cfg.encoder_cfg = SimpleNamespace()
    cfg.encoder_cfg.use_checkpoint = True

    print("Instantiating SwinEncoder3d...")
    m = SwinEncoder3d(
        cfg=cfg,
        use_checkpoint=cfg.encoder_cfg.use_checkpoint,
    ).eval()
    print("Instantiation successful.")

    n_params = count_parameters(m)
    print(f"\nModel: {type(m).__name__}")
    print("n_param: {:_}".format(n_params))

    x = torch.ones(2, cfg.in_chans, *cfg.roi_size)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        # The main forward pass now returns the features directly
        z = m(x)
        
    print("\nFeature shapes from each of the 4 stages (B, C, D, H, W):")
    for i, f in enumerate(z):
        print(f"  Stage {i+1}: {list(f.shape)}")