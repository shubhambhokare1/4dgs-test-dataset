_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 180]  # 6s * 30fps = 180 frames
    },
    bounds = 3.5  # sphere starts at x=-3.0, cube at origin
)

OptimizationParams = dict(
    densify_grad_threshold_coarse = 0.0001,
    densify_grad_threshold_fine_init = 0.0001,
    densify_grad_threshold_after = 0.0001,
)
