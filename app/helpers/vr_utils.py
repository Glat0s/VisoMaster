import os
import cv2
import numpy as np
import cupy as cp # Ensure visomaster environment has cupy
from PIL import Image
from cupyx.scipy import ndimage as cupy_ndi


# Assuming Equirec2Perspec_vr and Perspec2Equirec_vr are in app.processors.external
from app.processors.external.Equirec2Perspec_vr import Equirectangular as E2P_Equirectangular
from app.processors.external.Perspec2Equirec_vr import Perspective as P2E_Perspective

TEMP_DIR_VR = ".vr_temp_processing" # Define a temporary directory for VR processing files

class EquirectangularConverter:
    def __init__(self, equirect_image_data_rgb_uint8: np.ndarray):
        """
        Initializes with equirectangular image data.
        :param equirect_image_data_rgb_uint8: NumPy array (H, W, C) in RGB, uint8 format.
        """
        os.makedirs(TEMP_DIR_VR, exist_ok=True)
        self.equirect_image_bgr_uint8 = equirect_image_data_rgb_uint8[..., ::-1] # RGB to BGR
        self.height, self.width = self.equirect_image_bgr_uint8.shape[:2]
        # The E2P_Equirectangular class takes image data directly now
        self.e2p_instance = E2P_Equirectangular(self.equirect_image_bgr_uint8)

    def calculate_theta_phi_from_bbox(self, bbox_np: np.ndarray):
        x1, y1, x2, y2 = map(int, bbox_np)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        # Normalize coordinates: x to [-1, 1], y to [-1, 1] (from top to bottom)
        # Theta (longitude) from x, Phi (latitude) from y
        # Equirectangular width corresponds to 360 degrees, height to 180 degrees
        # For theta: 0 at center, -180 at left edge, +180 at right edge
        # For phi: 0 at equator (vertical center), +90 at top pole, -90 at bottom pole
        
        theta = (x_center / self.width - 0.5) * 360.0
        phi = -(y_center / self.height - 0.5) * 180.0 # Negative because image y is top-to-bottom

        return theta, phi

    def get_perspective_crop(self, FOV: float, THETA: float, PHI: float, height: int, width: int) -> np.ndarray:
        """
        Returns a perspective crop as a NumPy array (H, W, C) in BGR, uint8 format.
        """
        # GetPerspective returns a BGR numpy array
        persp_bgr = self.e2p_instance.GetPerspective(FOV, THETA, PHI, height, width)
        return persp_bgr


class PerspectiveConverter:
    def __init__(self, base_equirect_image_data_rgb_uint8: np.ndarray):
        """
        Initializes with the base equirectangular image data (used for dimensions and as background).
        :param base_equirect_image_data_rgb_uint8: NumPy array (H, W, C) in RGB, uint8 format.
        """
        os.makedirs(TEMP_DIR_VR, exist_ok=True)
        self.base_equirect_bgr_uint8 = base_equirect_image_data_rgb_uint8[..., ::-1].copy() # RGB to BGR
        self.orig_height, self.orig_width = self.base_equirect_bgr_uint8.shape[:2]
    
    def _apply_feathering(self, mask_cp: cp.ndarray) -> cp.ndarray:
        """ Applies feathering to a CuPy mask. Identical to vrswap's convert.py logic. """

        # Ensure mask_cp is float for gradient calculation
        mask_float_cp = mask_cp.astype(cp.float32)
        if mask_float_cp.ndim == 3 and mask_float_cp.shape[2] == 1:
            mask_float_cp = mask_float_cp.squeeze(axis=2) # Make it 2D if it's HxWx1

        mask_np = cp.asnumpy(mask_float_cp) # Convert to numpy for Sobel
        
        # Using a slightly larger ksize for Sobel might give a smoother gradient
        # and thus a wider feather, but ksize=3 is standard.
        gradient_x = cv2.Sobel(mask_np, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(mask_np, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude_np = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_magnitude_cp = cp.asarray(gradient_magnitude_np)

        # CV_64F implies float64, ensure mask_float_cp is float32 if that's sufficient
        gradient_x_cp = cupy_ndi.sobel(mask_float_cp, axis=1, mode='reflect') # dx, output is float32 if input is
        gradient_y_cp = cupy_ndi.sobel(mask_float_cp, axis=0, mode='reflect') # dy
        
        gradient_magnitude_cp = cp.sqrt(gradient_x_cp**2 + gradient_y_cp**2)
        
        max_grad = cp.amax(gradient_magnitude_cp)
        if max_grad < 1e-5: # Use a small epsilon to handle near-zero gradients
            # If no gradient, it means the mask is either all 0s or all 1s (or flat).
            # Return the original float mask.
            # If mask_float_cp was all zeros, this returns all zeros. If all ones, returns all ones.
            return mask_float_cp 

        feathered_mask_cp = 1.0 - gradient_magnitude_cp / max_grad
        return feathered_mask_cp


    def stitch_single_perspective(self,
                                  target_equirect_bgr: np.ndarray,
                                  processed_crop_bgr_uint8: np.ndarray,
                                  theta: float, phi: float, fov: float,
                                  is_left_eye: bool):
        """
        Stitches a single processed perspective crop back into the target equirectangular image.
        Modifies target_equirect_bgr in place.
        """

        p2e_instance = P2E_Perspective(processed_crop_bgr_uint8, FOV=fov, THETA=theta, PHI=phi)
        try:
            # GetEquirec returns NumPy arrays:
            # equirect_component_numpy: (H, W, C) BGR uint8, the processed crop warped to equirectangular space.
            # mask_numpy_original_shape: (H, W, 1) boolean, indicating valid warped pixels.
            equirect_component_numpy, mask_numpy_original_shape = p2e_instance.GetEquirec(self.orig_height, self.orig_width)
            # mask_cp_original_shape is likely (H, W) or (H, W, 1) and boolean
            # Make a copy for eye-specific masking
            equirect_component_np_bgr = equirect_component_numpy.copy()
            # Apply eye-specific masking (zero out the irrelevant half)
            half_width = self.orig_width // 2
            if is_left_eye:
                equirect_component_np_bgr[:, half_width:] = 0
            else: 
                equirect_component_np_bgr[:, :half_width] = 0
            
            equirect_component_cp = cp.asarray(equirect_component_np_bgr)
            mask_original_cp = cp.asarray(mask_numpy_original_shape) # This is (H,W,1) boolean CuPy array

            feathered_mask_cp_float = self._apply_feathering(mask_original_cp) # _apply_feathering expects a CuPy array
            
            target_equirect_cp = cp.asarray(target_equirect_bgr)

            if target_equirect_cp.dtype == cp.uint8:
                target_equirect_cp = target_equirect_cp.astype(cp.float32) / 255.0
            if equirect_component_cp.dtype == cp.uint8:
                equirect_component_cp = equirect_component_cp.astype(cp.float32) / 255.0

            # Ensure feathered_mask_cp_float is broadcastable for 3 channels (H, W, 1)
            if feathered_mask_cp_float.ndim == 2:
                feathered_mask_cp_float = feathered_mask_cp_float[..., cp.newaxis]
            
            # Ensure feathered_mask_cp_float is explicitly float32 for safety in arithmetic
            feathered_mask_cp_float = feathered_mask_cp_float.astype(cp.float32)
            composite_cp = target_equirect_cp * (1.0 - feathered_mask_cp_float) + \
                           equirect_component_cp * feathered_mask_cp_float
            
            if mask_original_cp.ndim == 3 and mask_original_cp.shape[2] == 1:
                mask_for_indexing_cp = mask_original_cp.squeeze(axis=2) 
            elif mask_original_cp.ndim == 2: 
                mask_for_indexing_cp = mask_original_cp 
            else:
                raise ValueError(f"Unexpected mask shape for indexing: {mask_original_cp.shape}")

            target_equirect_cp[mask_for_indexing_cp] = composite_cp[mask_for_indexing_cp]
            final_result_np = cp.asnumpy(cp.clip(target_equirect_cp * 255.0, 0, 255).astype(cp.uint8))
            
            # Explicitly delete CuPy arrays created in this scope
            del equirect_component_cp, mask_original_cp, feathered_mask_cp_float
            del target_equirect_cp, composite_cp
            if 'mask_for_indexing_cp' in locals(): # It might not be created if mask_original_cp is already 2D
                del mask_for_indexing_cp
                        
            target_equirect_bgr[:] = final_result_np
        finally:
            # Explicitly delete the instance to help with GPU memory cleanup
            if 'p2e_instance' in locals() and p2e_instance is not None:
                if hasattr(p2e_instance, '_img') and isinstance(p2e_instance._img, cp.ndarray):
                    del p2e_instance._img # Attempt to delete internal CuPy array
                del p2e_instance


def cleanup_temp_dir():
    import shutil
    if os.path.exists(TEMP_DIR_VR):
        shutil.rmtree(TEMP_DIR_VR)