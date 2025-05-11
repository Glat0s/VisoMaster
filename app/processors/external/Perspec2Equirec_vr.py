import cv2
import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
from functools import lru_cache

class Perspective:
    def __init__(self, img_name_or_data, FOV, THETA, PHI):

        if isinstance(img_name_or_data, str): 
            self._img = cv2.imread(img_name_or_data, cv2.IMREAD_COLOR) 
        elif isinstance(img_name_or_data, np.ndarray): 
            self._img = img_name_or_data # Assuming it's already BGR 
        else: 
            raise ValueError("Input must be a file path or a NumPy array.") 
 
        self._img = cp.asarray(self._img)
        self._width, self._height, _ = self._img.shape

        self._init_params(FOV, THETA, PHI)

    def _init_params(self, FOV, THETA, PHI):
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV
        self.w_len = cp.tan(cp.radians(self.wFOV / 2.0))
        self.h_len = cp.tan(cp.radians(self.hFOV / 2.0))

        self.R1, self.R2 = self._calc_rotation_matrices()

    @lru_cache(maxsize=None)
    def _calc_rotation_matrices(self):
        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        z_axis = cp.array([0.0, 0.0, 1.0], cp.float32)

        [R1, _] = cv2.Rodrigues(cp.asnumpy(z_axis * cp.radians(self.THETA)))
        [R2, _] = cv2.Rodrigues(cp.asnumpy(cp.dot(cp.asarray(R1), y_axis) * cp.radians(-self.PHI)))

        R1 = cp.asarray(cp.linalg.inv(cp.asarray(R1)))
        R2 = cp.asarray(cp.linalg.inv(cp.asarray(R2)))

        return R1, R2

    def SetParameters(self, FOV, THETA, PHI):
        self._init_params(FOV, THETA, PHI)

    def GetEquirec(self, height, width):
        x, y = cp.meshgrid(cp.linspace(-180, 180, width, dtype=cp.float32), cp.linspace(90, -90, height, dtype=cp.float32))

        xyz = cp.zeros((height, width, 3), dtype=cp.float32)
        xyz[..., 0] = cp.cos(cp.radians(x)) * cp.cos(cp.radians(y))
        xyz[..., 1] = cp.sin(cp.radians(x)) * cp.cos(cp.radians(y))
        xyz[..., 2] = cp.sin(cp.radians(y))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = cp.dot(self.R2, xyz)
        xyz = cp.dot(self.R1, xyz).T
        xyz = xyz.reshape([height, width, 3])
        #xyz /= xyz[..., 0, None]

        #conditions = (-self.w_len < xyz[..., 1]) & (xyz[..., 1] < self.w_len) & (-self.h_len < xyz[..., 2]) & (xyz[..., 2] < self.h_len)

        # Check if points are in front of the perspective camera's image plane.
        # xyz[..., 0] is the component along the camera's principal axis (depth).
        # Use a small epsilon to avoid issues with points exactly on the plane.
        is_in_front = xyz[..., 0] > 1e-5

        #lon_map = (xyz[..., 1] + self.w_len) / (2 * self.w_len) * self._width
        #lat_map = (-xyz[..., 2] + self.h_len) / (2 * self.h_len) * self._height

        # Initialize normalized screen coordinates (u, v) to a value that will be out of FOV.
        # Using cp.inf ensures they fail the w_len/h_len check if not properly updated.
        normalized_screen_x = cp.full_like(xyz[..., 1], cp.inf, dtype=cp.float32)
        normalized_screen_y = cp.full_like(xyz[..., 2], cp.inf, dtype=cp.float32)

        # Perform depth normalization only for points in front of the camera.
        # Safe divisor: xyz[..., 0] where it's in_front, 1.0 otherwise (to avoid div by zero/NaN).
        # The .squeeze() is important if xyz[..., 0, None] was used, but here xyz[..., 0] is already (H,W)
        safe_depth_divisor = cp.where(is_in_front, xyz[..., 0], 1.0)

        # Calculate normalized screen coordinates (u = x'/z', v = y'/z')
        normalized_screen_x = cp.where(is_in_front, xyz[..., 1] / safe_depth_divisor, cp.inf)
        normalized_screen_y = cp.where(is_in_front, xyz[..., 2] / safe_depth_divisor, cp.inf)

        # Check for NaN/Inf in normalized screen coordinates
        if cp.isnan(normalized_screen_x).any() or cp.isinf(normalized_screen_x).any() or \
           cp.isnan(normalized_screen_y).any() or cp.isinf(normalized_screen_y).any():
            # This case should ideally be rare due to cp.inf initialization and safe_depth_divisor
            # If it happens, force these to be out of FOV for safety
            normalized_screen_x = cp.where(cp.isfinite(normalized_screen_x), normalized_screen_x, cp.inf)
            normalized_screen_y = cp.where(cp.isfinite(normalized_screen_y), normalized_screen_y, cp.inf)

        # Conditions for being within FOV, using the normalized screen coordinates
        fov_conditions = cp.isfinite(normalized_screen_x) & cp.isfinite(normalized_screen_y) & \
                         (-self.w_len < normalized_screen_x) & \
                         (normalized_screen_x < self.w_len) & \
                         (-self.h_len < normalized_screen_y) & \
                         (normalized_screen_y < self.h_len)

        # The final mask: must be in front of camera AND within its FOV
        conditions = is_in_front & fov_conditions

        # Map these normalized screen coordinates to pixel coordinates in the perspective image
        lon_map = (normalized_screen_x + self.w_len) / (2 * self.w_len) * self._width
        lat_map = (-normalized_screen_y + self.h_len) / (2 * self.h_len) * self._height

        # Ensure lat_map and lon_map are finite where conditions are true.
        # Where conditions are false, map_coordinates will use 0,0 which should be fine if self._img is valid there.
        # Or, more robustly, map to a known safe coordinate or handle fill_value in map_coordinates if available.
        # For now, we rely on `persp *= mask` later.
        safe_lat_map = cp.where(conditions & cp.isfinite(lat_map), lat_map, 0.0)
        safe_lon_map = cp.where(conditions & cp.isfinite(lon_map), lon_map, 0.0)
        coordinates = cp.stack([safe_lat_map, safe_lon_map], axis=0).astype(cp.float32)

        # Explicitly delete intermediate large CuPy arrays to free GPU memory sooner
        del x, y, xyz, safe_depth_divisor
        del normalized_screen_x, normalized_screen_y
        del lon_map, lat_map, safe_lat_map, safe_lon_map
        # 'conditions' is still needed for the mask

        persp = cp.empty((height, width, self._img.shape[2]), dtype=self._img.dtype)
        for i in range(self._img.shape[2]):
            ndi.map_coordinates(self._img[..., i], coordinates, output=persp[..., i], order=1, mode='nearest')

        del coordinates # Delete after use

        mask = conditions[..., cp.newaxis]  # Compute mask
        persp *= mask  # Apply mask to persp
        
        del conditions # Delete after use

        return cp.asnumpy(persp), cp.asnumpy(mask)

    def resetDevice():
        #device = cuda.get_current_device()
        device = cp.cuda.get_current_device()
        device.reset()