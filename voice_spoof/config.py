"""
Model configs for AASIST / AASIST-L, copied from the official
clovaai/aasist repo's config/*.conf files (model_config section only).

nb_samp = 64600 -> ~4.04 sec at 16kHz. This is the fixed window length
the model was trained on; our inference wrapper pads/trims chunks to this.
"""

AASIST_L_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

AASIST_FULL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

CHECKPOINTS = {
    "aasist-l": "checkpoints/AASIST-L.pth",
    "aasist": "checkpoints/AASIST.pth",
}

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 64600  # fixed input length the model expects

# Thresholds for turning the raw 0-1 score into a human-readable label.
# Tune these once you've validated against real speech + real spoofed clips -
# these starting values are reasonable defaults, not validated ground truth.
RISK_THRESHOLDS = {
    "medium": 0.5,
    "high": 0.8,
}
