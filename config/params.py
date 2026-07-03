MAX_INTENSITY = 5 # The maximum intensity for each AU
BINARY_OCCURRENCE_THRESHOLD = 2 # An intensity of higher than or equal to 2 is considered an occurrence in binary occurrence detection
INVALID_INDICATOR_IE = MAX_INTENSITY + 1 # The indicator of a frame with invalid human expert label in AU intensity estimation
INVALID_INDICATOR_D = 2 # The indicator of a frame with invalid human expert label in AU detection

# The preprocessing steps for each frame
PREPROCESS_STEPS = [('cvtColor', 'BGR2RGB'),
                    ('cropping', ('mediapipe', {'ymin': -0.2, 'ymax': 0.05, 'xmin': -0.125, 'xmax': 0.125})),
                    ('alignment', 112),
                    ('heNlm', (0.5, 0.5)),
                    ('padding', None),
                    ('cvtColor', 'RGB2BGR')]

# The length of each frame
IMAGE_LENGTH = 112

# The steps of transforming each frame in deep learning model training
TRANSFORM_STEPS = [('BGR2RGB', None),
                   ('RandomHorizontalFlip', None),
                   ('RandomResizedCrop', {'size': IMAGE_LENGTH, 'scale': (0.90, 1), 'ratio': (19/20, 20/19)}),
                   ('Resize', IMAGE_LENGTH)]

# Other settings in model training
SEED = 42
OPTIMIZER_SETTINGS = {'base_lr': 1e-5, 'last_layer_lr': 1e-4, 'weight_decay': 5e-4}
CLASS_WEIGHT_METHOD = 'inverse'
CLASS_WEIGHT_GROUPS = [[0, 1], [2, 3, 4, 5]]
LOSS_WEIGHTS = [1, 1, 1]

