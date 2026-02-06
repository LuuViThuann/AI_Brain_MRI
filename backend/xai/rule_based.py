"""
rule_based.py
Rule-based analysis for brain tumor diagnosis.

Provides:
  - Statistical measurements (area, volume, ratio)
  - Location detection
  - Risk level classification
  - Quantitative features extraction
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple


class RuleBasedAnalyzer:
    """
    Analyze brain MRI using statistical rules and thresholds.
    100% explainable baseline for comparison with AI.
    """
    
    # Risk thresholds (based on medical literature)
    RISK_THRESHOLDS = {
        'low': 100,        # < 100mm² = low risk
        'medium': 500,     # 100-500mm² = medium risk
        'high': 500        # > 500mm² = high risk
    }
    
    # Location risk factors
    CRITICAL_LOCATIONS = [
        'frontal',    # Motor control
        'temporal',   # Language, memory
        'brainstem'   # Vital functions
    ]

     # WHO Grading References (simplified for LGG)
    WHO_THRESHOLDS = {
        'grade_1': {
            'max_diameter_mm': 30,
            'description': 'Pilocytic astrocytoma - typically well-defined'
        },
        'grade_2': {
            'max_diameter_mm': 60,
            'description': 'Diffuse astrocytoma - infiltrative growth pattern'
        },
        'grade_3': {
            'max_diameter_mm': float('inf'),
            'description': 'Anaplastic astrocytoma - rapid growth'
        }
    }
     # BraTS Dataset Statistics (for reference)
    BRATS_STATISTICS = {
        'mean_tumor_volume_mm3': 45000,
        'median_tumor_volume_mm3': 32000,
        'mean_tumor_area_mm2': 2500,
        'percentile_25': 1200,
        'percentile_75': 4800
    }
    
    def __init__(self, pixel_to_mm=0.5):
        """
        Args:
            pixel_to_mm: Conversion factor from pixels to mm
                        (default: 0.5mm per pixel for typical MRI)
        """
        self.pixel_to_mm = pixel_to_mm
    
    def analyze(self, mask: np.ndarray, mri_image: np.ndarray = None) -> Dict:
        """
        Complete rule-based analysis.
        
        Args:
            mask: Binary segmentation mask (256x256)
            mri_image: Original MRI image (256x256), optional
        
        Returns:
            {
                "tumor_area_mm2": float,
                "tumor_volume_mm3": float,
                "tumor_ratio": float,
                "location": str,
                "risk_level": "Low|Medium|High",
                "rules_triggered": [str],
                "quantitative_features": dict,
                "warnings": [str]
            }
        """
        # Ensure mask is binary
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Basic measurements
        tumor_pixels = np.sum(mask_binary)
        total_pixels = mask_binary.size
        
        tumor_area_mm2 = tumor_pixels * (self.pixel_to_mm ** 2)
        tumor_ratio = tumor_pixels / total_pixels
        
        # Volume estimation (assume slice thickness ~5mm)
        slice_thickness = 5.0
        tumor_volume_mm3 = tumor_area_mm2 * slice_thickness
        
        # Location detection
        location = self._detect_location(mask_binary)
        
        # Risk assessment
        risk_level, rules = self._assess_risk(
            tumor_area_mm2, tumor_ratio, location, mask_binary
        )
        
        # Quantitative features
        features = self._extract_features(mask_binary, mri_image)
        
        # Warnings
        warnings = self._generate_warnings(
            tumor_area_mm2, tumor_ratio, location, features
        )

        medical_context = self._get_medical_context(tumor_area_mm2, features)
        
        # 2. BraTS percentile ranking
        brats_percentile = self._calculate_brats_percentile(tumor_area_mm2)
        
        # 3. Enhanced risk with references
        risk_level, rules, risk_rationale = self._assess_risk_enhanced(
            tumor_area_mm2, tumor_ratio, location, mask_binary, features
        )
        
        
        return {
            # Existing fields
            "tumor_area_mm2": round(tumor_area_mm2, 2),
            "tumor_volume_mm3": round(tumor_volume_mm3, 2),
            "tumor_ratio": round(tumor_ratio * 100, 2),
            "location": location,
            "risk_level": risk_level,
            "rules_triggered": rules,
            "quantitative_features": features,
            "warnings": warnings,
            
            # ===== NEW FIELDS =====
            "medical_context": medical_context,
            "dataset_comparison": {
                "brats_percentile": brats_percentile,
                "comparison": self._compare_to_dataset(tumor_area_mm2)
            },
            "risk_rationale": risk_rationale,
            "standards_reference": {
                "who_grading": "Simplified WHO CNS tumor classification",
                "dataset": "BraTS (Brain Tumor Segmentation Challenge)",
                "threshold_source": "Medical literature + dataset statistics"
            }
        }

    def _get_medical_context(self, area_mm2: float, features: Dict) -> Dict:
            """Provide medical context based on WHO standards."""
            
            # Estimate diameter from area (assuming roughly circular)
            diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
            
            # WHO grade estimation (for LGG only - Grade 1-2)
            if diameter_mm < self.WHO_THRESHOLDS['grade_1']['max_diameter_mm']:
                who_grade_est = "Grade I (estimated)"
                description = self.WHO_THRESHOLDS['grade_1']['description']
            else:
                who_grade_est = "Grade II (estimated)"
                description = self.WHO_THRESHOLDS['grade_2']['description']
            
            return {
                "estimated_diameter_mm": round(diameter_mm, 1),
                "who_grade_estimate": who_grade_est,
                "tumor_characteristics": description,
                "size_category": self._get_size_category(area_mm2),
                "disclaimer": "Estimation based on size only. Histopathology required for definitive grading."
            }

    def _get_size_category(self, area_mm2: float) -> str:
            """Categorize tumor size relative to clinical significance."""
            if area_mm2 < 100:
                return "Very small (<100mm²)"
            elif area_mm2 < 500:
                return "Small (100-500mm²)"
            elif area_mm2 < 2000:
                return "Medium (500-2000mm²)"
            elif area_mm2 < 5000:
                return "Large (2000-5000mm²)"
            else:
                return "Very large (>5000mm²)"
    
    def _calculate_brats_percentile(self, area_mm2: float) -> int:
        """Calculate percentile rank compared to BraTS dataset."""
        if area_mm2 < self.BRATS_STATISTICS['percentile_25']:
            return 25
        elif area_mm2 < self.BRATS_STATISTICS['mean_tumor_area_mm2']:
            return 50
        elif area_mm2 < self.BRATS_STATISTICS['percentile_75']:
            return 75
        else:
            return 90
    
    def _compare_to_dataset(self, area_mm2: float) -> str:
        """Provide comparison narrative."""
        percentile = self._calculate_brats_percentile(area_mm2)
        
        if percentile <= 25:
            return "Smaller than 75% of tumors in BraTS dataset"
        elif percentile <= 50:
            return "Below median size in BraTS dataset"
        elif percentile <= 75:
            return "Above median, within typical range"
        else:
            return "Larger than 75% of tumors in BraTS dataset"
    
    def _assess_risk_enhanced(
        self,
        area_mm2: float,
        ratio: float,
        location: str,
        mask: np.ndarray,
        features: Dict
    ) -> Tuple[str, List[str], Dict]:
        """
        Enhanced risk assessment with rationale.
        
        Returns:
            (risk_level, rules_triggered, rationale_dict)
        """
        rules = []
        risk_score = 0
        rationale_points = []
        
        # Rule 1: Size-based risk (with medical context)
        if area_mm2 < 100:
            rules.append(f"Very small tumor (<100mm²) - {self._calculate_brats_percentile(area_mm2)}th percentile")
            risk_score += 1
            rationale_points.append("Tumor size is small relative to clinical significance")
        elif area_mm2 < 500:
            rules.append(f"Small-medium tumor (100-500mm²) - {self._calculate_brats_percentile(area_mm2)}th percentile")
            risk_score += 2
            rationale_points.append("Tumor size is moderate, requires monitoring")
        else:
            rules.append(f"Large tumor (>500mm²) - {self._calculate_brats_percentile(area_mm2)}th percentile")
            risk_score += 3
            rationale_points.append("Tumor size exceeds typical low-grade glioma range")
        
        # Rule 2: Brain coverage (with threshold reference)
        coverage_threshold = 0.05  # 5% - based on clinical significance
        if ratio > coverage_threshold:
            rules.append(f"Significant brain coverage (>{coverage_threshold*100:.1f}% of slice)")
            risk_score += 2
            rationale_points.append(f"Tumor occupies >{coverage_threshold*100:.0f}% of brain slice - clinically significant")
        
        # Rule 3: Location risk (with functional region context)
        location_lower = location.lower()
        location_risk_added = False
        for critical_loc in self.CRITICAL_LOCATIONS:
            if critical_loc in location_lower:
                functional_impact = self._get_functional_impact(critical_loc)
                rules.append(f"Critical location: {critical_loc} - {functional_impact}")
                risk_score += 2
                rationale_points.append(f"Location in {critical_loc} region may affect: {functional_impact}")
                location_risk_added = True
                break
        
        if not location_risk_added:
            rationale_points.append("Location in non-eloquent cortex")
        
        # Rule 4: Shape irregularity (growth pattern indicator)
        circularity = features.get('circularity', 1.0)
        if circularity < 0.5:
            rules.append(f"Irregular shape (circularity: {circularity:.2f}) - suggests infiltrative growth")
            risk_score += 1
            rationale_points.append("Irregular borders suggest diffuse/infiltrative growth pattern")
        
        # Determine final risk level
        if risk_score <= 2:
            risk_level = "Low"
        elif risk_score <= 5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        rationale = {
            "risk_score": risk_score,
            "max_score": 9,
            "factors_considered": rationale_points,
            "classification_method": "Additive risk scoring based on size, location, coverage, and morphology",
            "interpretation": f"Risk score {risk_score}/9 classified as {risk_level} risk"
        }
        
        return risk_level, rules, rationale
    
    def _get_functional_impact(self, location: str) -> str:
        """Get functional impact description for brain region."""
        impact_map = {
            'frontal': 'Motor control, executive function, speech (Broca\'s area)',
            'temporal': 'Memory formation, language comprehension (Wernicke\'s area), hearing',
            'parietal': 'Sensory processing, spatial awareness, calculation',
            'occipital': 'Visual processing',
            'brainstem': 'Vital functions (breathing, heart rate, consciousness)'
        }
        return impact_map.get(location, 'Various cognitive/motor functions')

    
    def _detect_location(self, mask: np.ndarray) -> str:
        """
        Detect tumor location based on centroid position.
        
        Brain regions (simplified):
          - Frontal: top-front (y < 0.4, x any)
          - Temporal: sides (y 0.4-0.7, x < 0.3 or x > 0.7)
          - Parietal: top-back (y < 0.4, x 0.4-0.6)
          - Occipital: back (y any, x > 0.7)
        """
        # Find centroid
        M = cv2.moments(mask)
        
        if M['m00'] == 0:
            return "unknown"
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Normalize to 0-1 range
        h, w = mask.shape
        cx_norm = cx / w
        cy_norm = cy / h
        
        # Determine hemisphere
        hemisphere = "left" if cx_norm < 0.5 else "right"
        
        # Determine lobe
        if cy_norm < 0.4:
            if cx_norm < 0.4 or cx_norm > 0.6:
                lobe = "frontal"
            else:
                lobe = "parietal"
        elif cy_norm < 0.7:
            if cx_norm < 0.3 or cx_norm > 0.7:
                lobe = "temporal"
            else:
                lobe = "central"
        else:
            if cx_norm > 0.7:
                lobe = "occipital"
            else:
                lobe = "inferior"
        
        return f"{hemisphere} {lobe}"
    
    def _assess_risk(
        self,
        area_mm2: float,
        ratio: float,
        location: str,
        mask: np.ndarray
    ) -> Tuple[str, List[str]]:
        """
        Assess risk level based on multiple factors.
        
        Returns:
            (risk_level, rules_triggered)
        """
        rules = []
        risk_score = 0
        
        # Rule 1: Size-based risk
        if area_mm2 < self.RISK_THRESHOLDS['low']:
            rules.append("Small tumor (<100mm²)")
            risk_score += 1
        elif area_mm2 < self.RISK_THRESHOLDS['medium']:
            rules.append("Medium tumor (100-500mm²)")
            risk_score += 2
        else:
            rules.append("Large tumor (>500mm²)")
            risk_score += 3
        
        # Rule 2: Brain coverage ratio
        if ratio > 0.05:  # >5% of slice
            rules.append("Significant brain coverage (>5%)")
            risk_score += 2
        
        # Rule 3: Location-based risk
        location_lower = location.lower()
        for critical_loc in self.CRITICAL_LOCATIONS:
            if critical_loc in location_lower:
                rules.append(f"Near critical region: {critical_loc}")
                risk_score += 2
                break
        
        # Rule 4: Multiple tumor regions
        num_components = self._count_tumor_regions(mask)
        if num_components > 1:
            rules.append(f"Multiple tumor regions ({num_components})")
            risk_score += 2
        
        # Rule 5: Irregular shape
        circularity = self._calculate_circularity(mask)
        if circularity < 0.5:
            rules.append("Irregular tumor shape")
            risk_score += 1
        
        # Determine final risk level
        if risk_score <= 2:
            risk_level = "Low"
        elif risk_score <= 5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return risk_level, rules
    
    def _extract_features(
        self,
        mask: np.ndarray,
        mri_image: np.ndarray = None
    ) -> Dict:
        """Extract quantitative features."""
        features = {}
        
        # Geometric features
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['area_pixels'] = int(area)
            features['perimeter_pixels'] = round(perimeter, 2)
            
            # Circularity (4π*area / perimeter²)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                features['circularity'] = round(circularity, 3)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['bbox'] = [int(x), int(y), int(w), int(h)]
            features['aspect_ratio'] = round(w / h if h > 0 else 0, 3)
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                features['solidity'] = round(solidity, 3)
        
        # Intensity features (if MRI image provided)
        if mri_image is not None:
            tumor_pixels = mri_image[mask > 0]
            if len(tumor_pixels) > 0:
                features['mean_intensity'] = round(float(np.mean(tumor_pixels)), 3)
                features['std_intensity'] = round(float(np.std(tumor_pixels)), 3)
                features['min_intensity'] = float(np.min(tumor_pixels))
                features['max_intensity'] = float(np.max(tumor_pixels))
        
        return features
    
    def _count_tumor_regions(self, mask: np.ndarray) -> int:
        """Count number of disconnected tumor regions."""
        num_labels, _ = cv2.connectedComponents(mask)
        return num_labels - 1  # Subtract background
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate tumor circularity (0-1, 1=perfect circle)."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)
    
    def _generate_warnings(
        self,
        area_mm2: float,
        ratio: float,
        location: str,
        features: Dict
    ) -> List[str]:
        """Generate clinical warnings based on findings."""
        warnings = []
        
        # Size warnings
        if area_mm2 > 1000:
            warnings.append("⚠ Very large tumor detected (>1000mm²)")
        
        # Coverage warnings
        if ratio > 0.1:
            warnings.append("⚠ Extensive brain involvement (>10% of slice)")
        
        # Location warnings
        location_lower = location.lower()
        if 'frontal' in location_lower:
            warnings.append("⚠ Frontal lobe involvement may affect motor control")
        elif 'temporal' in location_lower:
            warnings.append("⚠ Temporal lobe involvement may affect language/memory")
        
        # Shape warnings
        if features.get('circularity', 1.0) < 0.4:
            warnings.append("⚠ Highly irregular shape detected")
        
        # Multiple regions warning
        if features.get('solidity', 1.0) < 0.7:
            warnings.append("⚠ Irregular boundaries suggest infiltrative growth")
        
        return warnings


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    print("=" * 70)
    print("  Rule-Based Analyzer - Test")
    print("=" * 70)
    print()
    
    # Create synthetic tumor mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (100, 90), 40, 1, -1)  # Circular tumor
    
    # Create synthetic MRI
    mri = np.random.randint(50, 150, (256, 256), dtype=np.uint8)
    mri[mask > 0] = np.random.randint(150, 200, np.sum(mask > 0))
    
    # Analyze
    analyzer = RuleBasedAnalyzer(pixel_to_mm=0.5)
    result = analyzer.analyze(mask, mri)
    
    # Print results
    print("Analysis Results:")
    print(f"  Tumor Area: {result['tumor_area_mm2']} mm²")
    print(f"  Tumor Volume: {result['tumor_volume_mm3']} mm³")
    print(f"  Brain Coverage: {result['tumor_ratio']}%")
    print(f"  Location: {result['location']}")
    print(f"  Risk Level: {result['risk_level']}")
    print()
    
    print("Rules Triggered:")
    for rule in result['rules_triggered']:
        print(f"  • {rule}")
    print()
    
    print("Quantitative Features:")
    for key, value in result['quantitative_features'].items():
        print(f"  {key}: {value}")
    print()
    
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  {warning}")
    
    print("\n✅ Test complete")