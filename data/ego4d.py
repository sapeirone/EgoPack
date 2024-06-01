from typing import Dict, Literal

Ego4dBackbones = Literal["omnivore_image_swinl", "omnivore_video_swinl"]


# From https://ego4d-data.org/docs/data/features/
__features_window_sizes__: Dict[Ego4dBackbones, int] = {
    "omnivore_image_swinl": 1,
    "omnivore_video_swinl": 32,
}

# From https://ego4d-data.org/docs/data/features/
__features_strides__: Dict[Ego4dBackbones, int] = {
    "omnivore_image_swinl": 5,
    "omnivore_video_swinl": 16,
}

__features_sizes__: Dict[Ego4dBackbones, int] = {
    "omnivore_image_swinl": 1536,
    "omnivore_video_swinl": 1536,
}
