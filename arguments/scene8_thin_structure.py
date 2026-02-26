_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 150]  # 5s * 30fps = 150 frames
    },
    bounds = 2.5  # moving rod travels from x=-2.0 to x=2.0
)

OptimizationParams = dict(
    # Very low thresholds: thin rods (radius 0.02) produce small gradients
    # and need aggressive densification to be properly reconstructed
    densify_grad_threshold_coarse = 0.00005,
    densify_grad_threshold_fine_init = 0.00005,
    densify_grad_threshold_after = 0.00005,
)
