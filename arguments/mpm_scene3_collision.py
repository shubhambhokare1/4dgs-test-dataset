# 4DGaussians config for the MPM port of scene 3 (mpm/), any camera subset.
_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 90]  # 3 s * 30 fps (+1) = 91 frames
    },
    bounds = 3.5  # balls start ~2.6 m from the centre and rebound outwards
)

OptimizationParams = dict(
    densify_grad_threshold_coarse = 0.0001,
    densify_grad_threshold_fine_init = 0.0001,
    densify_grad_threshold_after = 0.0001,
)

ModelParams = dict(
    white_background = False,  # images are opaque (white sky + floor), so this has no effect
)
