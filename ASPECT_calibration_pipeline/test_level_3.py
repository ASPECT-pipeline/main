from level_3.modules.collect_data import resave_ASPECT_transmission
import level_3.main_composition as main_composition
from tqdm import tqdm

"""
python3 ASPECT_calibration_pipeline/test_level_3.py
"""

def generate_aspect_transmission():
    resave_ASPECT_transmission()

# generate_aspect_transmission()

def train_composition_models():
    for _ in tqdm(range(10)):
        y_pred = main_composition.pipeline()

# train_composition_models()