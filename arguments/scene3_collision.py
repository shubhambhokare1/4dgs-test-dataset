_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 150]  # 5s * 30fps = 150 frames
    },
    bounds = 2.5  # balls start at ~2.0 units from center
)

OptimizationParams = dict(
    # Lower thresholds so small balls (radius 0.15) trigger densification
    densify_grad_threshold_coarse = 0.0001,
    densify_grad_threshold_fine_init = 0.0001,
    densify_grad_threshold_after = 0.0001,
)