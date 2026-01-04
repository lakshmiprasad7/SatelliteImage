"""
Scene Analysis - Disabled (Pure DL Mode)
Classical CV methods not allowed
"""
import numpy as np

class SceneAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, image: np.ndarray) -> dict:
        """
        Scene analysis disabled in pure DL mode
        Returns minimal metadata
        """
        analysis = {
            'scene_type': 'unknown',
            'has_tall_structures': False,
            'has_smooth_surfaces': False,
            'illumination_direction': None,
            'shadow_geometry': None,
            'detected_objects': []
        }
        
        return analysis
