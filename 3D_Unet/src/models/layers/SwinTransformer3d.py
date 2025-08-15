from types import SimpleNamespace
import torch
import torch.nn as nn
from torchvision.models.video import swin3d_b
from torchvision.models.video.swin_transformer import SwinTransformer3d
from timm.models._manipulate import checkpoint
import os

class SwinEncoder3d(SwinTransformer3d):
    """
    A 3D Swin Transformer encoder that extracts multi-scale features.
    """
    def __init__(
        self, 
        cfg: SimpleNamespace, 
        use_checkpoint: bool = False,
        inference_mode: bool = False
    ):
        # Initialize the parent SwinTransformer3d class with its default parameters
        new_patch_size = [32, 4, 4] 
        super().__init__(
            patch_size=new_patch_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=[8, 7, 7], 
            stochastic_depth_prob=0.3, # Default for swin3d_b
        )
        self.use_checkpoint = use_checkpoint

        # --- Architectural Adaptation ---
        # This part MUST run regardless of inference_mode to ensure the model's
        # input layer matches the desired number of channels (cfg.in_chans).
        new_proj = nn.Conv3d(
            cfg.in_chans, 128, kernel_size=new_patch_size, stride=new_patch_size
        )

        # --- Weight Loading controlled by `inference_mode` ---
        if not inference_mode:
            print("Loading pretrained Swin3D-B weights...")
            # This block loads and adapts pretrained weights.
            
            # 1. Load the state dict from file
            wpath = "/argusdata4/naamagav/byu/model_zoo/swin3d_b_1k-24f7c7c6.pth"
            if not os.path.exists(wpath):
                 print(f"Warning: Weight file not found at {wpath}")
                 state_dict = {}
            else:
                 state_dict = torch.load(wpath)

            # 2. Get the state dict of a fresh, uninitialized swin3d_b model
            #    to ensure all keys are present.
            vanilla_model = swin3d_b(weights=None)
            original_model_weights = vanilla_model.state_dict()
            original_model_weights.update(state_dict)

            # 3. Remove the weight and bias for the original patch embedding layer.
            original_patch_embed_weight = original_model_weights.pop("patch_embed.proj.weight")
            original_patch_embed_bias = original_model_weights.pop("patch_embed.proj.bias")
            
            # 4. Load all other weights that perfectly match
            self.load_state_dict(original_model_weights, strict=False)

            # 5. Adapt the loaded weights for our new projection layer
            new_weight = original_patch_embed_weight.mean(dim=2, keepdim=True).repeat(1, 1, new_patch_size[0], 1, 1)
            
            if cfg.in_chans != 3:
                new_weight = new_weight.mean(dim=1, keepdim=True).repeat(1, cfg.in_chans, 1, 1, 1)
            
            new_proj.weight.data = new_weight
            new_proj.bias.data = original_patch_embed_bias

        else:
            print("Skipping weight loading due to inference_mode=True. Model has random initialization.")

        # --- Final Assignment ---
        # Replace the model's default patch_embed projection layer with our new one.
        self.patch_embed.proj = new_proj

        # Determine the output channels from each stage for the decoder
        with torch.no_grad():
            dummy_input = torch.randn(1, cfg.in_chans, *cfg.roi_size)
            features = self.forward(dummy_input)
            self.channels = [f.shape[1] for f in features]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Overrides the default forward pass to return a list of feature maps.
        """
        res = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Iterate through the flat list of stages and downsamplers
        for i, layer in enumerate(self.features):
            x = checkpoint(layer, x) if self.use_checkpoint and self.training else layer(x)
            
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

    print("--- Instantiating SwinEncoder3d with inference_mode=False (loading weights) ---")
    m_trained = SwinEncoder3d(
        cfg=cfg,
        use_checkpoint=cfg.encoder_cfg.use_checkpoint,
        inference_mode=False,
    ).eval()
    print("Instantiation successful.")

    print("\n--- Instantiating SwinEncoder3d with inference_mode=True (skipping weights) ---")
    m_scratch = SwinEncoder3d(
        cfg=cfg,
        use_checkpoint=cfg.encoder_cfg.use_checkpoint,
        inference_mode=True,
    ).eval()
    print("Instantiation successful.")


    n_params = count_parameters(m_trained)
    print(f"\nModel: {type(m_trained).__name__}")
    print(f"Trainable Parameters: {n_params:_}")

    x = torch.randn(2, cfg.in_chans, *cfg.roi_size)
    print(f"\nInput shape: {x.shape}")
    
    with torch.inference_mode():
        z = m_trained(x)
        
    print("\nFeature shapes from each of the 4 stages (B, C, D, H, W):")
    for i, f in enumerate(z):
        print(f"  Stage {i+1}: {list(f.shape)}")

