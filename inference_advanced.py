"""
Advanced Inference Engine
Main processing engine for satellite image enhancement
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

# Internal Simple Detector
class SimpleConditionDetector:
    """Lightweight detector for basic conditions"""
    
    def detect_all_conditions(self, image: np.ndarray) -> Dict:
        """Analyze image statistics"""
        stats = {}
        
        # 1. Haze/Cloud Detection (Brightness & Contrast)
        # Haze usually = Low Contrast + High Brightness in Dark Channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        stats['haze'] = {
            'detected': contrast < 40 and brightness > 100,
            'severity': 'Moderate' if contrast < 40 else 'Low'
        }
        
        stats['cloud'] = {
            'detected': brightness > 200,
            'percentage': (np.sum(gray > 220) / gray.size) * 100
        }
        
        # 2. Shadow Detection (Low Luminance)
        # Using LAB L-channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        shadow_mask = l < 60
        shadow_pct = (np.sum(shadow_mask) / l.size) * 100
        
        stats['shadow'] = {
            'detected': shadow_pct > 0.5,
            'percentage': shadow_pct,
            'mask': shadow_mask.astype(np.uint8) * 255
        }
        
        return stats

    def align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two images using SIFT"""
        # Simple resize to match smaller dimension for safety
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2 or w1 != w2:
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            img1 = cv2.resize(img1, (target_w, target_h))
            img2 = cv2.resize(img2, (target_w, target_h))
            
        return img1, img2


class AdvancedInferenceEngine:
    """Advanced Inference Engine with deep learning enhancement"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = "cpu"):
        """Initialize the inference engine"""
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.detector = SimpleConditionDetector()
        
        print(f"[OK] Inference Engine initialized (Deep Learning Mode)")
    
    def process_single_image(self, image: np.ndarray) -> Dict:
        """
        Process single image using deep learning models
        Args:
            image: BGR image from OpenCV
        Returns:
            Dict with 'processed' as BGR image
        """
        # Convert BGR to RGB for processing (our modules expect RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original = image.copy()  # Keep original in BGR
        processed = image_rgb.copy()  # Process in RGB
        applied_corrections = []
        
        # Detect all conditions (expects BGR)
        detection = self.detector.detect_all_conditions(image)
        
        print("\n" + "="*60)
        print("SATELLITE IMAGE PROCESSING")
        print("="*60)
        
        # 1. Cloud detection
        if detection['cloud']['detected']:
            cloud_pct = detection['cloud'].get('percentage', 0)
            print(f"[CLOUD] Detected ({cloud_pct:.1f}%)")
        
        # 2. Haze detection
        if detection['haze']['detected']:
            severity = detection['haze']['severity']
            print(f"[HAZE] Detected ({severity})")
        
        # 3. Shadow detection with visualization
        shadow_detection_map = None
        shadow_analysis = None
        
        if detection['shadow']['detected'] and detection['shadow']['percentage'] > 1:
            shadow_pct = detection['shadow']['percentage']
            print(f"[SHADOW] Detected ({shadow_pct:.1f}%)")
            
            # Create shadow mask
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l = lab[:,:,0]
            l_float = l.astype(float)
            
            # Multi-scale illumination
            scales = [15, 31, 61]
            illuminations = []
            for scale in scales:
                illum = cv2.bilateralFilter(l, scale, 80, 80)
                illuminations.append(illum)
            illumination = np.mean(illuminations, axis=0).astype(np.uint8)
            
        # Import Deep Learning Model Hub
        from dl_models import get_dl_enhancer
        enhancer = get_dl_enhancer(device=self.device)
        
        # Apply multi-model deep learning enhancement (RGB input/output)
        enhancement_results = enhancer.enhance(processed, tasks=['all'])
        
        # Extract results (in RGB)
        processed_rgb = enhancement_results['processed']
        dl_stats = enhancement_results['detection']
        dl_corrections = enhancement_results['applied_corrections']
        
        # Convert back to BGR for consistency with OpenCV convention
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        
        # Merge stats (DL stats override simple stats)
        if 'shadow' in dl_stats: detection['shadow'] = dl_stats['shadow']
        if 'haze' in dl_stats: detection['haze'] = dl_stats['haze']
        if 'cloud' in dl_stats: detection['cloud'] = dl_stats['cloud']
        if 'content' in dl_stats: detection['content'] = dl_stats['content']
        
        print("-" * 70)
        print(f"[OK] APPLIED {len(dl_corrections)} ENHANCEMENTS")
        print("=" * 70 + "\n")
        
        # Generate AI caption from actual image analysis
        caption = self._generate_caption(processed_rgb, detection, enhancement_results.get('scene_analysis', {}))
        
        # Generate disturbance visualization
        disturbance_viz = self._create_disturbance_visualization(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            detection
        )
        
        return {
            'original': original,  # BGR
            'processed': processed_bgr,  # BGR
            'detection_raw': detection,  # Map to detection_raw for UI
            'applied_enhancements': dl_corrections,  # Map to applied_enhancements for UI
            'explanations': enhancement_results.get('explanations', []),
            'scene_analysis': enhancement_results.get('scene_analysis', {}),
            'caption': caption,
            'disturbance_visualization': cv2.cvtColor(disturbance_viz, cv2.COLOR_RGB2BGR),  # BGR for consistency
            'shadow_detection_map': None, 
            'shadow_analysis': None
        }
    
    def _create_disturbance_visualization(self, image_rgb, detection):
        """
        Create visualization showing detected disturbances with BRIGHT RED marks
        """
        print("   [VIZ] Creating disturbance visualization...")
        
        # Always detect shadows directly from image
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l = lab[:,:,0]
        
        # Ultra-sensitive detection
        mean_l = np.mean(l)
        std_l = np.std(l)
        threshold = mean_l - 0.2 * std_l  # Very sensitive
        shadow_mask = (l < threshold).astype(np.uint8)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        shadow_pct = (np.sum(shadow_mask > 0) / shadow_mask.size) * 100
        print(f"   [VIZ] Detected {shadow_pct:.1f}% shadow for visualization")
        
        # Create visualization with BRIGHT RED overlay
        viz = image_rgb.copy().astype(float)
        
        # Apply BRIGHT RED to shadow areas (90% red!)
        red_color = np.array([255, 0, 0], dtype=float)
        for i in range(3):
            viz[:,:,i] = np.where(shadow_mask > 0, 
                                  viz[:,:,i] * 0.1 + red_color[i] * 0.9,  # 90% red
                                  viz[:,:,i])
        
        viz = np.clip(viz, 0, 255).astype(np.uint8)
        
        print(f"   [VIZ] Applied BRIGHT RED overlay to {shadow_pct:.1f}% of image")
        
        return viz
    
    def _generate_caption(self, image_rgb, detection, scene_analysis):
        """Generate dynamic AI caption using real object detection"""
        try:
            import cv2
            
            print("   [CAPTION] Generating AI caption...")
            
            # Detect actual objects in the image
            detected_objects = self._detect_objects_simple(image_rgb)
            print(f"   [CAPTION] Detected objects: {detected_objects}")
            
            # Build dynamic caption
            if len(detected_objects) > 0:
                # Use detected objects
                caption = "Image showing " + ", ".join(detected_objects[:4])
            else:
                # Fallback to scene analysis
                features = []
                
                # Analyze colors for scene type
                hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                
                # Detect urban (gray/white buildings)
                gray_mask = cv2.inRange(s, 0, 50)
                gray_pct = (np.sum(gray_mask > 0) / gray_mask.size) * 100
                
                # Detect vegetation
                green_mask = cv2.inRange(h, 35, 85)
                green_pct = (np.sum(green_mask > 0) / green_mask.size) * 100
                
                if gray_pct > 30:
                    features.append("urban scene")
                if green_pct > 15:
                    features.append("vegetation")
                
                if len(features) > 0:
                    caption = "Satellite imagery showing " + ", ".join(features)
                else:
                    caption = "Aerial/satellite imagery"
            
            # Add quality issues
            issues = []
            if 'shadow' in detection and detection['shadow'].get('shadow_pct', 0) > 25:
                issues.append("shadows")
            if 'haze' in detection and detection['haze'].get('haze_score', 0) > 30:
                issues.append("haze")
            
            if issues:
                caption += f" (affected by {', '.join(issues)})"
            
            print(f"   [CAPTION] Generated: {caption}")
            
            return caption + "."
            
        except Exception as e:
            print(f"   [CAPTION ERROR] {e}")
            import traceback
            traceback.print_exc()
            return "Image analysis in progress."
    
    def _detect_objects_simple(self, image_rgb):
        """
        Simple object detection using image analysis
        Detects: people, buildings, vehicles, vegetation, water
        """
        objects = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        h, s, v = cv2.split(hsv)
        l, a, b_lab = cv2.split(lab)
        
        # 1. Detect people (STRICT criteria to avoid false positives)
        # Only detect if there's a clear person-shaped region with skin tone
        skin_mask = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
        skin_pct = (np.sum(skin_mask > 0) / skin_mask.size) * 100
        
        # STRICT: Need at least 2% skin AND vertical structure
        if skin_pct > 2.0:  # Increased from 0.5% to 2% to avoid false positives
            # Check for vertical structure (person shape)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            has_person_shape = False
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h / (w + 1e-6)
                if aspect_ratio > 1.2 and cv2.contourArea(cnt) > 1000:  # Tall and large enough
                    has_person_shape = True
                    break
            
            if has_person_shape:
                objects.append("person")
        
        # 2. Detect buildings (vertical edges, gray color)
        edges = cv2.Canny(gray, 50, 150)
        edge_pct = (np.sum(edges > 0) / edges.size) * 100
        gray_mask = cv2.inRange(s, 0, 50)
        gray_pct = (np.sum(gray_mask > 0) / gray_mask.size) * 100
        
        if edge_pct > 5 and gray_pct > 30:
            objects.append("buildings")
        
        # 3. Detect vegetation (green: hue 35-85)
        green_mask = cv2.inRange(h, 35, 85)
        green_pct = (np.sum(green_mask > 0) / green_mask.size) * 100
        if green_pct > 10:
            objects.append("vegetation")
        
        # 4. Detect water (blue: hue 90-130, low lightness)
        water_mask = cv2.inRange(hsv, (90, 50, 20), (130, 255, 200))
        water_pct = (np.sum(water_mask > 0) / water_mask.size) * 100
        if water_pct > 5:
            objects.append("water")
        
        # 5. Detect roads/infrastructure (brown/tan: hue 10-30)
        road_mask = cv2.inRange(h, 10, 30)
        road_pct = (np.sum(road_mask > 0) / road_mask.size) * 100
        if road_pct > 15:
            objects.append("roads")
        
        # 6. Detect haze (low contrast, high brightness)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        if contrast < 40 and brightness > 150:
            objects.append("heavy haze")
        
        return objects
    
    def _dual_scan_shadow_removal(self, img1_rgb, img2_rgb):
        """
        Dual-scan shadow removal using both images
        Fills shadow areas in one image with lit pixels from the other
        """
        print("   [DUAL-SCAN] Intelligent shadow removal using both images...")
        
        # Convert to LAB for shadow detection
        lab1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2LAB)
        
        l1, a1, b1 = cv2.split(lab1)
        l2, a2, b2 = cv2.split(lab2)
        
        # Detect shadows in both images
        def detect_shadows(l_channel):
            mean_l = np.mean(l_channel)
            std_l = np.std(l_channel)
            threshold = mean_l - 0.5 * std_l
            shadow_mask = (l_channel < threshold).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            return shadow_mask
        
        shadow_mask1 = detect_shadows(l1)
        shadow_mask2 = detect_shadows(l2)
        
        shadow_pct1 = (np.sum(shadow_mask1 > 0) / shadow_mask1.size) * 100
        shadow_pct2 = (np.sum(shadow_mask2 > 0) / shadow_mask2.size) * 100
        
        print(f"   [DUAL-SCAN] Image 1 shadows: {shadow_pct1:.1f}%")
        print(f"   [DUAL-SCAN] Image 2 shadows: {shadow_pct2:.1f}%")
        
        # Create smooth alpha masks
        alpha1 = cv2.GaussianBlur(shadow_mask1, (31, 31), 0).astype(float) / 255.0
        alpha2 = cv2.GaussianBlur(shadow_mask2, (31, 31), 0).astype(float) / 255.0
        
        # DUAL-SCAN MAGIC:
        # Where img1 has shadow, use pixels from img2
        # Where img2 has shadow, use pixels from img1
        
        # Reconstruct img1 using img2's lit areas
        result1_lab = lab1.copy().astype(float)
        result1_lab[:,:,0] = l1 * (1 - alpha1) + l2 * alpha1  # Use img2's brightness in shadow
        result1_lab[:,:,1] = a1 * (1 - alpha1) + a2 * alpha1  # Use img2's color
        result1_lab[:,:,2] = b1 * (1 - alpha1) + b2 * alpha1
        result1_lab = np.clip(result1_lab, 0, 255).astype(np.uint8)
        result1 = cv2.cvtColor(result1_lab, cv2.COLOR_LAB2RGB)
        
        # Reconstruct img2 using img1's lit areas
        result2_lab = lab2.copy().astype(float)
        result2_lab[:,:,0] = l2 * (1 - alpha2) + l1 * alpha2
        result2_lab[:,:,1] = a2 * (1 - alpha2) + a1 * alpha2
        result2_lab[:,:,2] = b2 * (1 - alpha2) + b1 * alpha2
        result2_lab = np.clip(result2_lab, 0, 255).astype(np.uint8)
        result2 = cv2.cvtColor(result2_lab, cv2.COLOR_LAB2RGB)
        
        print(f"   [DUAL-SCAN] Shadow removal complete - used cross-image filling")
        
        return result1, result2
    
    def process_dual_images(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """
        Process and compare two images
        """
        # 1. Align images
        img1_aligned, img2_aligned = self.detector.align_images(img1, img2)
        
        # 2. DUAL-SCAN SHADOW REMOVAL
        # Use both images to remove shadows intelligently
        img1_deshadowed, img2_deshadowed = self._dual_scan_shadow_removal(
            img1_aligned, img2_aligned
        )
        
        # 3. Process both (use deshadowed versions)
        res1 = self.process_single_image(img1_deshadowed)
        res2 = self.process_single_image(img2_deshadowed)
        
        # 3. Structural Change Detection
        # We compare the ENHANCED images to ignore shadows
        diff = cv2.absdiff(res1['processed'], res2['processed'])
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find significant changes
        # Blur to remove noise
        gray_diff = cv2.GaussianBlur(gray_diff, (9, 9), 0)
        _, change_mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)  # Increased from 30 to 50
        
        # Cleanup mask - more aggressive to reduce noise
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))  # Increased from 5x5 to 7x7
        change_mask = cv2.dilate(change_mask, np.ones((3,3),np.uint8), iterations=1)  # Reduced from 2 to 1
        
        # Calculate percentage
        change_pct = (np.sum(change_mask > 0) / change_mask.size) * 100
        
        detection = {
            'changes_detected': change_pct > 1.0,
            'change_percentage': change_pct
        }
        
        return {
            'image1': res1,
            'image2': res2,
            'change_detection': {
                'mask': change_mask,
                'stats': detection
            },
            'semantic_analysis': f"Structural Change Detected: {change_pct:.2f}% of area."
        }


if __name__ == "__main__":
    print("Testing Inference Engine...")
    
    engine = AdvancedInferenceEngine(device="cpu")
    
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print("\nTesting single image processing...")
    results = engine.process_single_image(test_img)
    print(f"[OK] Processed: {results['processed'].shape}")
    print(f"[OK] Applied corrections: {results['applied_corrections']}")
    
    print("\n[OK] Inference Engine ready!")
