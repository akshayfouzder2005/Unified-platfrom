"""
Image preprocessing utilities for marine species identification
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Advanced image preprocessing for marine species identification
    """
    
    def __init__(self):
        self.target_size = (224, 224)  # Standard for most CNN models
        self.enhance_contrast = True
        self.enhance_brightness = True
        
    def preprocess_fish_image(
        self, 
        image: Union[np.ndarray, Image.Image, str], 
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess fish images for species identification
        
        Args:
            image: Input image (array, PIL Image, or file path)
            target_size: Target size for resizing (width, height)
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            if image is None:
                raise ValueError("Could not load image")
                
            # Set target size
            if target_size is None:
                target_size = self.target_size
                
            # Basic preprocessing steps
            processed_image = self._enhance_image_quality(image)
            processed_image = self._remove_background_noise(processed_image)
            processed_image = self._resize_image(processed_image, target_size)
            processed_image = self._normalize_image(processed_image)
            
            logger.info(f"Successfully preprocessed image to shape: {processed_image.shape}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def preprocess_otolith_image(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Specialized preprocessing for otolith images
        
        Args:
            image: Input otolith image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed otolith image
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            elif isinstance(image, Image.Image):
                image = np.array(image.convert('L'))
            
            if image is None:
                raise ValueError("Could not load otolith image")
            
            if target_size is None:
                target_size = self.target_size
            
            # Otolith-specific preprocessing
            processed_image = self._enhance_otolith_contrast(image)
            processed_image = self._detect_otolith_edges(processed_image)
            processed_image = self._resize_image(processed_image, target_size)
            processed_image = self._normalize_image(processed_image)
            
            # Convert to 3 channels for model compatibility
            if len(processed_image.shape) == 2:
                processed_image = np.stack([processed_image] * 3, axis=-1)
            
            logger.info(f"Successfully preprocessed otolith image to shape: {processed_image.shape}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing otolith image: {str(e)}")
            raise
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality with contrast and brightness adjustments"""
        try:
            pil_image = Image.fromarray(image)
            
            # Enhance contrast
            if self.enhance_contrast:
                contrast_enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = contrast_enhancer.enhance(1.2)
            
            # Enhance brightness
            if self.enhance_brightness:
                brightness_enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = brightness_enhancer.enhance(1.1)
            
            return np.array(pil_image)
        except Exception as e:
            logger.warning(f"Could not enhance image quality: {str(e)}")
            return image
    
    def _remove_background_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove background noise using Gaussian blur and edge detection"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Apply bilateral filter for edge-preserving smoothing
            filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
            
            return filtered
        except Exception as e:
            logger.warning(f"Could not remove background noise: {str(e)}")
            return image
    
    def _enhance_otolith_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast specifically for otolith images"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            # Apply morphological operations to enhance shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Could not enhance otolith contrast: {str(e)}")
            return image
    
    def _detect_otolith_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect and enhance otolith edges for better shape analysis"""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            
            # Combine original image with edge information
            enhanced = cv2.addWeighted(image, 0.7, edges, 0.3, 0)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Could not detect otolith edges: {str(e)}")
            return image
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio"""
        try:
            height, width = image.shape[:2]
            target_width, target_height = target_size
            
            # Calculate scaling factor to maintain aspect ratio
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Create canvas with target size
            if len(image.shape) == 3:
                canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=np.uint8)
            else:
                canvas = np.zeros((target_height, target_width), dtype=np.uint8)
            
            # Center the resized image on canvas
            start_y = (target_height - new_height) // 2
            start_x = (target_width - new_width) // 2
            canvas[start_y:start_y + new_height, start_x:start_x + new_width] = resized
            
            return canvas
        except Exception as e:
            logger.warning(f"Could not resize image: {str(e)}")
            # Fallback to simple resize
            return cv2.resize(image, target_size)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1] range"""
        try:
            normalized = image.astype(np.float32) / 255.0
            return normalized
        except Exception as e:
            logger.warning(f"Could not normalize image: {str(e)}")
            return image
    
    def extract_fish_features(self, image: np.ndarray) -> dict:
        """
        Extract morphological features from fish images
        
        Args:
            image: Preprocessed fish image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Extract basic morphological features
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assuming it's the fish)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate basic features
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h
                
                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Moments for shape analysis
                moments = cv2.moments(largest_contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = 0, 0
                
                features.update({
                    'area': area,
                    'perimeter': perimeter,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'centroid_x': cx,
                    'centroid_y': cy,
                    'bounding_width': w,
                    'bounding_height': h
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting fish features: {str(e)}")
            return {}
    
    def batch_preprocess(self, images: list, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess multiple images in batch
        
        Args:
            images: List of images to preprocess
            target_size: Target size for all images
            
        Returns:
            Batch of preprocessed images as numpy array
        """
        try:
            if not images:
                return np.array([])
            
            processed_batch = []
            for image in images:
                processed = self.preprocess_fish_image(image, target_size)
                processed_batch.append(processed)
            
            return np.array(processed_batch)
            
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {str(e)}")
            raise