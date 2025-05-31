import threading
import os
import subprocess as sp
import gc
import traceback
from typing import Dict, TYPE_CHECKING

from packaging import version
import numpy as np
import onnxruntime
import torch
import onnx
from torchvision.transforms import v2
from PySide6 import QtCore
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("No TensorRT Found")
    TENSORRT_AVAILABLE = False

from app.processors.utils.engine_builder import onnx_to_trt as onnx2trt
from app.processors.utils.tensorrt_predictor import TensorRTPredictor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import models_list, arcface_mapping_model_dict, models_trt_list, models_dir
from app.processors.utils import faceutil # Assuming faceutil contains create_faded_inner_mask
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
lock = threading.Lock()

class ModelsProcessor(QtCore.QObject):
    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession

    def __init__(self, main_window: 'MainWindow', device='cuda'):
        super().__init__()
        self.main_window = main_window
        self.provider_name = 'TensorRT'
        self.device = device
        self.model_lock = threading.RLock()  # Reentrant lock for model access
        self.trt_ep_options = {
            # 'trt_max_workspace_size': 3 << 30,  # Dimensione massima dello spazio di lavoro in bytes
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': "tensorrt-engines",
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': "tensorrt-engines",
            'trt_dump_ep_context_model': True,
            'trt_ep_context_file_path': "tensorrt-engines",
            'trt_layer_norm_fp32_fallback': True,
            'trt_builder_optimization_level': 5,
        }
        self.providers = [
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]       
        self.nThreads = 2
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)

        # Initialize models and models_path
        self.models: Dict[str, onnxruntime.InferenceSession] = {}
        self.models_path = {}
        self.models_data = {}
        for model_data in models_list:
            model_name, model_path = model_data['model_name'], model_data['local_path']
            self.models[model_name] = None #Model Instance
            self.models_path[model_name] = model_path
            self.models_data[model_name] = {'local_path': model_data['local_path'], 'hash': model_data['hash'], 'url': model_data.get('url')}

        self.dfm_models: Dict[str, DFMModel] = {}

        if TENSORRT_AVAILABLE:
            # Initialize models_trt and models_trt_path
            self.models_trt = {}
            self.models_trt_path = {}
            for model_data in models_trt_list:
                model_name, model_path = model_data['model_name'], model_data['local_path']
                self.models_trt[model_name] = None #Model Instance
                self.models_trt_path[model_name] = model_path

        self.face_detectors = FaceDetectors(self)
        self.face_landmark_detectors = FaceLandmarkDetectors(self)
        self.face_masks = FaceMasks(self)
        self.face_restorers = FaceRestorers(self)
        self.face_swappers = FaceSwappers(self)
        self.frame_enhancers = FrameEnhancers(self)
        self.face_editors = FaceEditors(self)

        # Denoiser specific initializations
        self.lp_mask_crop_latent = faceutil.create_faded_inner_mask(size=(64, 64), border_thickness=3, fade_thickness=8, blur_radius=3, device=self.device)
        self.lp_mask_crop_latent = torch.unsqueeze(self.lp_mask_crop_latent, 0) # Shape: [1, 64, 64]
        self.betas_np = np.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=np.float64)**2 # Common linear schedule for 1000 steps
        self.alphas_np = 1.0 - self.betas_np
        self.alphas_cumprod_np = np.cumprod(self.alphas_np, axis=0)
        self.alphas_cumprod_torch = torch.from_numpy(self.alphas_cumprod_np).float().to(self.device)
        self.vae_scale_factor = 0.18215 # Typical LDM VAE scale factor

        self.clip_session = []
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        self.mean_lmk = []
        self.anchors  = []
        self.emap = []
        self.LandmarksSubsetIdxs = [
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

        self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/1.0, 1/1.0, 1/1.0 ])
        
        self.lp_mask_crop = self.face_editors.lp_mask_crop
        self.lp_lip_array = self.face_editors.lp_lip_array

    def load_model(self, model_name, session_options=None):
        with self.model_lock:
            self.main_window.model_loading_signal.emit()
            # QApplication.processEvents()
            # if not is_file_exists(self.models_path[model_name]):
            #     download_file(model_name, self.models_path[model_name], self.models_data[model_name]['hash'], self.models_data[model_name]['url'])
            if session_options is None:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], providers=self.providers)
            else:
                model_instance = onnxruntime.InferenceSession(self.models_path[model_name], sess_options=session_options, providers=self.providers)

            # Check if another thread has already loaded an instance for this model, if yes then delete the current one and return that instead
            if self.models[model_name]:
                del model_instance
                gc.collect()
                return self.models[model_name]
            self.main_window.model_loaded_signal.emit()

            return model_instance

    def load_dfm_model(self, dfm_model):
        with self.model_lock:
            if not self.dfm_models.get(dfm_model):
                self.main_window.model_loading_signal.emit()
                max_models_to_keep = self.main_window.control['MaxDFMModelsSlider']
                total_loaded_models = len(self.dfm_models)
                if total_loaded_models==max_models_to_keep:
                    print("Clearing DFM Model")
                    model_name, model_instance = list(self.dfm_models.items())[0]
                    del model_instance
                    self.dfm_models.pop(model_name)
                    gc.collect()
                try:
                    self.dfm_models[dfm_model] = DFMModel(self.main_window.dfm_models_data[dfm_model], self.providers, self.device)
                except:
                    traceback.print_exc()   
                    self.dfm_models[dfm_model] = None         
                self.main_window.model_loaded_signal.emit()
            return self.dfm_models[dfm_model]


    def load_model_trt(self, model_name, custom_plugin_path=None, precision='fp16', debug=False):
        # self.showModelLoadingProgressBar()
        #time.sleep(0.5)
        self.main_window.model_loading_signal.emit()

        if not os.path.exists(self.models_trt_path[model_name]):
            onnx2trt(onnx_model_path=self.models_path[model_name],
                     trt_model_path=self.models_trt_path[model_name],
                     precision=precision,
                     custom_plugin_path=custom_plugin_path,
                     verbose=False
                    )
        model_instance = TensorRTPredictor(model_path=self.models_trt_path[model_name], custom_plugin_path=custom_plugin_path, pool_size=self.nThreads, device=self.device, debug=debug)

        self.main_window.model_loaded_signal.emit()
        return model_instance

    def delete_models(self):
        for model_name, model_instance in self.models.items():
            del model_instance
            self.models[model_name] = None
        self.clip_session = []
        gc.collect()

    def delete_models_trt(self):
        if TENSORRT_AVAILABLE:
            for model_data in models_trt_list:
                model_name = model_data['model_name']
                if isinstance(self.models_trt[model_name], TensorRTPredictor):
                    # È un'istanza di TensorRTPredictor
                    self.models_trt[model_name].cleanup()
                    del self.models_trt[model_name]
                    self.models_trt[model_name] = None #Model Instance
            gc.collect()

    def delete_models_dfm(self):
        keys_to_remove = []
        for model_name, model_instance in self.dfm_models.items():
            del model_instance
            keys_to_remove.append(model_name)
        
        for model_name in keys_to_remove:
            self.dfm_models.pop(model_name)
        
        self.clip_session = []
        gc.collect()

    def unload_model(self, model_name_to_unload):
        with self.model_lock:
            if model_name_to_unload in self.models and self.models[model_name_to_unload] is not None:
                print(f"Unloading model: {model_name_to_unload}")
                del self.models[model_name_to_unload]
                self.models[model_name_to_unload] = None # Explicitly set to None after del
                gc.collect()
                torch.cuda.empty_cache()
            # else:
            #     print(f"Model {model_name_to_unload} not found or not loaded for unloading.")

    def showModelLoadingProgressBar(self):
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                                ('TensorrtExecutionProvider', self.trt_ep_options),
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
                if version.parse(trt.__version__) < version.parse("10.2.0") and provider_name == "TensorRT-Engine":
                    print("TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0.")
                    provider_name = "TensorRT"

            case "CPU":
                providers = [
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cpu'
            case "CUDA":
                providers = [
                                ('CUDAExecutionProvider'),
                                ('CPUExecutionProvider')
                            ]
                self.device = 'cuda'
            #case _:

        self.providers = providers
        self.provider_name = provider_name
        self.lp_mask_crop = self.lp_mask_crop.to(self.device)

        return self.provider_name

    def set_number_of_threads(self, value):
        self.nThreads = value
        self.delete_models_trt()

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]
    
    def clear_gpu_memory(self):
        self.delete_models()
        self.delete_models_dfm()
        self.delete_models_trt()
        torch.cuda.empty_cache()


    def load_inswapper_iss_emap(self, model_name):
        with self.model_lock:
            if not self.models[model_name]:
                self.main_window.model_loading_signal.emit()
                graph = onnx.load(self.models_path[model_name]).graph
                self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                self.main_window.model_loaded_signal.emit()

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(img, detect_mode, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles)
    
    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        return self.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss, detect_mode, score, from_points)

    def get_arcface_model(self, face_swapper_model): 
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(self, img, kps, similarity_type='Opal', arcface_model='Inswapper128ArcFace'):
        return self.face_swappers.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        self.face_swappers.run_inswapper(image, embedding, output)

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        return self.face_swappers.calc_swapper_latent_iss(source_embedding, version)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        self.face_swappers.run_iss_swapper(image, embedding, output, version)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GhostFace-v2'):
        self.face_swappers.run_swapper_ghostface(image, embedding, output, swapper_model)

    def calc_swapper_latent_cscs(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    def run_enhance_frame_tile_process(self, img, enhancer_type, tile_size=256, scale=1):
        return self.frame_enhancers.run_enhance_frame_tile_process(img, enhancer_type, tile_size, scale)

    def run_deoldify_artistic(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)

    def run_deoldify_stable(self, image, output):
        return self.frame_enhancers.run_deoldify_artistic(image, output)
    
    def run_deoldify_video(self, image, output):
        return self.frame_enhancers.run_deoldify_video(image, output)
    
    def run_ddcolor_artistic(self, image, output):
        return self.frame_enhancers.run_ddcolor_artistic(image, output)

    def run_ddcolor(self, tensor_gray_rgb, output_ab):
        return self.frame_enhancers.run_ddcolor(tensor_gray_rgb, output_ab)

    def run_occluder(self, image, output):
        self.face_masks.run_occluder(image, output)

    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def run_faceparser(self, image, output):
        self.face_masks.run_faceparser(image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)
    
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(kp_source, eye_close_ratio, face_editor_type)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(kp_source, lip_close_ratio, face_editor_type)

    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        return self.face_editors.lp_warp_decode(feature_3d, kp_source, kp_driving, face_editor_type)

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100-cos_dist*50

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        return self.face_restorers.apply_facerestorer(swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score)

    def apply_occlusion(self, img, amount):
        return self.face_masks.apply_occlusion(img, amount)
    
    def apply_dfl_xseg(self, img, amount):
        return self.face_masks.apply_dfl_xseg(img, amount)
    
    def apply_face_parser(self, img, parameters):
        return self.face_masks.apply_face_parser(img, parameters)
    
    def apply_face_makeup(self, img, parameters):
        return self.face_editors.apply_face_makeup(img, parameters)
    
    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        return self.face_masks.restore_mouth(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset)

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        return self.face_masks.restore_eyes(img_orig, img_swap, kpss_orig, blend_alpha, feather_radius, size_factor, radius_factor_x, radius_factor_y, x_offset, y_offset, eye_spacing_offset)

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        return self.face_masks.apply_fake_diff(swapped_face, original_face, DiffAmount)

    def run_vae_encoder(self, image_input_tensor: torch.Tensor, output_latent_tensor: torch.Tensor):
        """
        Runs the VAE encoder model.
        image_input_tensor: Batch x 3 x Height x Width, float32, normalized to [-1, 1]
        output_latent_tensor: Placeholder for Batch x 8 x LatentH x LatentW, float32
        """
        model_name = 'RefLDMVAEEncoder'
        if not self.models[model_name]:
            # Temporarily force CUDA EP for this model if TensorRT is causing issues
            if self.provider_name.startswith("TensorRT"):
                print(f"DEBUG: Forcing CUDAExecutionProvider for {model_name} due to TensorRT issues.")
                temp_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                try:
                    self.models[model_name] = onnxruntime.InferenceSession(
                        self.models_path[model_name], providers=temp_providers
                    )
                except Exception as e:
                    print(f"Error loading {model_name} with CUDA EP, falling back to default load: {e}")
                    self.models[model_name] = self.load_model(model_name) # Fallback to default load
            else:
                self.models[model_name] = self.load_model(model_name) # Fallback to default load

            io_binding = self.models[model_name].io_binding()
            io_binding.bind_input(name='image_input', device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(image_input_tensor.shape), buffer_ptr=image_input_tensor.data_ptr())
            print(f"DEBUG: run_vae_encoder - output_latent_tensor shape for binding: {output_latent_tensor.shape}") # ADD THIS
            io_binding.bind_output(name='latent_pre_quant_unscaled', device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(output_latent_tensor.shape), buffer_ptr=output_latent_tensor.data_ptr())

            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device != "cpu":
                self.syncvec.cpu()
            self.models[model_name].run_with_iobinding(io_binding)

    def run_ref_ldm_unet(self, unet_filename: str,
                         x_noisy_plus_lq_latent: torch.Tensor,
                         timesteps_tensor: torch.Tensor,
                         context_tensor: torch.Tensor, # Added
                         class_labels_tensor: torch.Tensor, # Added
                         is_ref_flag_tensor: torch.Tensor, # Added
                         output_unet_tensor: torch.Tensor):
        """
        Runs the UNet denoiser model.
        unet_filename: The filename of the UNet ONNX model (e.g., "ref_ldm_unet_n1.onnx").
        x_noisy_plus_lq_latent: Batch x 16 x LatentH x LatentW, float32
        timesteps_tensor: Batch, int64
        context_tensor: Dummy context, e.g., Batch x 1 x 1, float32
        class_labels_tensor: Dummy class labels, e.g., Batch, int64
        is_ref_flag_tensor: Scalar boolean tensor, False for denoising.
        output_unet_tensor: Placeholder for Batch x 8 x LatentH x LatentW, float32
        """
        
        model_name = unet_filename # Use the filename as the key for the self.models dictionary

        if not self.models.get(model_name) or self.models[model_name] is None: # Check for None as well
            model_path_to_load = os.path.join(models_dir, unet_filename)
            if not os.path.exists(model_path_to_load):
                print(f"Error: UNet Denoiser model file not found: {model_path_to_load}")
                # Optionally, raise an error or handle it by returning/not processing
                return
            print(f"Loading UNet Denoiser: {model_path_to_load}")
            self.main_window.model_loading_signal.emit()
            try:
                self.models[model_name] = onnxruntime.InferenceSession(model_path_to_load, providers=self.providers)
            except Exception as e:
                print(f"Error loading ONNX model {model_path_to_load}: {e}")
                self.main_window.model_loaded_signal.emit() # Ensure dialog is hidden
                return # Cannot proceed
            self.main_window.model_loaded_signal.emit()

        ort_session = self.models[model_name]
        model_inputs = ort_session.get_inputs()
        model_outputs = ort_session.get_outputs()

        io_binding = ort_session.io_binding()

        # Bind inputs dynamically using names from the loaded model
        # Assumes the order of tensors passed to this function matches the ONNX model's input order
        # if the model has fewer inputs than expected, it will only bind the ones that exist.

        io_binding.bind_input(name=model_inputs[0].name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(x_noisy_plus_lq_latent.shape), buffer_ptr=x_noisy_plus_lq_latent.data_ptr())
        io_binding.bind_input(name=model_inputs[1].name, device_type=self.device, device_id=0, element_type=np.int64, shape=tuple(timesteps_tensor.shape), buffer_ptr=timesteps_tensor.data_ptr())

        if len(model_inputs) > 2: # Corresponds to context_tensor
            io_binding.bind_input(name=model_inputs[2].name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(context_tensor.shape), buffer_ptr=context_tensor.data_ptr())
        if len(model_inputs) > 3: # Corresponds to class_labels_tensor
            io_binding.bind_input(name=model_inputs[3].name, device_type=self.device, device_id=0, element_type=np.int64, shape=tuple(class_labels_tensor.shape), buffer_ptr=class_labels_tensor.data_ptr())
        if len(model_inputs) > 4: # Corresponds to is_ref_flag_tensor
            io_binding.bind_input(name=model_inputs[4].name, device_type=self.device, device_id=0, element_type=np.bool_, shape=tuple(is_ref_flag_tensor.shape), buffer_ptr=is_ref_flag_tensor.data_ptr())

        io_binding.bind_output(name=model_outputs[0].name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(output_unet_tensor.shape), buffer_ptr=output_unet_tensor.data_ptr())
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models[model_name].run_with_iobinding(io_binding)

    def run_vae_decoder(self, latent_input_tensor: torch.Tensor, output_image_tensor: torch.Tensor):
        """
        Runs the VAE decoder model.
        latent_input_tensor: Batch x 8 x LatentH x LatentW, float32 (expected to be unscaled by vae_scale_factor as per denoiser usage)
        output_image_tensor: Placeholder for Batch x 3 x H x W, float32, normalized to [-1, 1]
        """
        model_name = 'RefLDMVAEDecoder'
        if not self.models[model_name]:
            self.models[model_name] = self.load_model(model_name)

        io_binding = self.models[model_name].io_binding()
        io_binding.bind_input(name='scaled_latent_input', device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(latent_input_tensor.shape), buffer_ptr=latent_input_tensor.data_ptr()) # Assuming ONNX node name is 'scaled_latent_input'
        io_binding.bind_output(name='image_output', device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(output_image_tensor.shape), buffer_ptr=output_image_tensor.data_ptr())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        self.models[model_name].run_with_iobinding(io_binding)

    def apply_denoiser_unet(self, 
                            image_cxhxw_uint8: torch.Tensor, 
                            unet_filename: str,
                            denoiser_mode: str = "Single Step (Fast)", # New parameter
                            denoiser_single_step_t: int = 10,       # Default t for single step
                            frame_number_for_seed: int = 0         # For seeding
                            ) -> torch.Tensor:
        # Input: CxHxW, uint8, RGB, range [0, 255]
        # Output: CxHxW, uint8, RGB, range [0, 255]
        # unet_filename: The specific UNet model file to use.

        if not unet_filename or unet_filename == "No UNet models found":
            # print(f"Denoiser: No UNet model selected or available ('{unet_filename}'). Skipping denoise pass.")
            return image_cxhxw_uint8 # Return original image if no model

        unet_model_path = os.path.join(models_dir, unet_filename)

        # If the selected model doesn't exist, try a default fallback
        if not os.path.exists(unet_model_path):
            print(f"Denoiser: Selected UNet model file '{unet_model_path}' not found.")
            default_unet_fallback = "ref_ldm_unet_n1.onnx" # A common default, adjust if needed
            default_unet_path = os.path.join(models_dir, default_unet_fallback)
            if os.path.exists(default_unet_path):
                print(f"Denoiser: Attempting to use default fallback UNet model: {default_unet_fallback}")
                unet_filename = default_unet_fallback # Switch to default for this run
            else:
                print(f"Denoiser: Default fallback UNet model '{default_unet_path}' also not found. Skipping denoise pass.")
                return image_cxhxw_uint8

        # The UNet expects a 64x64 latent. Assuming VAE f=8, input image should be 512x512.
        target_proc_dim = 512 # Denoiser components are trained for 512x512 inputs
        _, h_input, w_input = image_cxhxw_uint8.shape
        # Attempt to force deterministic algorithms for the scope of denoiser operations
        old_deterministic_state = torch.are_deterministic_algorithms_enabled()
        #torch.use_deterministic_algorithms(True)
        
        if h_input != target_proc_dim or w_input != target_proc_dim:
            print(f"DEBUG: Denoiser - Resizing input face from {h_input}x{w_input} to {target_proc_dim}x{target_proc_dim}")
            resize_transform = v2.Resize((target_proc_dim, target_proc_dim), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
            image_to_process_cxhxw_uint8 = resize_transform(image_cxhxw_uint8)
        else:
            image_to_process_cxhxw_uint8 = image_cxhxw_uint8

        # Set seed for deterministic VAE encoding for this frame
        # This is crucial if the VAE itself has any stochastic behavior or unseeded random initializations
        torch.manual_seed(frame_number_for_seed)
        # torch.use_deterministic_algorithms(True) # Potentially uncomment if issues persist, might impact performance

        h_proc, w_proc = image_to_process_cxhxw_uint8.shape[1], image_to_process_cxhxw_uint8.shape[2]

        # 1. Normalize image to [-1, 1] and add batch dimension
        image_normalized_bchw = (image_to_process_cxhxw_uint8.float() / 127.5) - 1.0
        #image_normalized_bchw = image_normalized_bchw.unsqueeze(0)
        # print(f"DEBUG: Denoiser - image_normalized_bchw min: {image_normalized_bchw.min():.4f}, max: {image_normalized_bchw.max():.4f}, mean: {image_normalized_bchw.mean():.4f}")
        image_normalized_bchw = image_normalized_bchw.unsqueeze(0).contiguous()


        # 2. VAE Encode
        # Latent dimensions should be 64x64 for the UNet
        latent_h = h_proc // 8 # Should be 64 if h_proc is 512
        latent_w = w_proc // 8 # Should be 64 if w_proc is 512

        # encoded_latent_8_channel is z_lq (unscaled)
        # VAE Encoder outputs 8 channels for latent_pre_quant_unscaled
        encoded_latent_8_channel = torch.empty((1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device).contiguous()

        # --- VAE Encoder Call ---
        vae_enc_model_name = 'RefLDMVAEEncoder'
        if not self.models.get(vae_enc_model_name) or self.models[vae_enc_model_name] is None:
            self.models[vae_enc_model_name] = self.load_model(vae_enc_model_name)
        
        vae_enc_session = self.models[vae_enc_model_name]
        vae_enc_input_name = vae_enc_session.get_inputs()[0].name
        vae_enc_output_name = vae_enc_session.get_outputs()[0].name
        
        vae_enc_io_binding = vae_enc_session.io_binding()
        vae_enc_io_binding.bind_input(name=vae_enc_input_name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(image_normalized_bchw.shape), buffer_ptr=image_normalized_bchw.data_ptr())
        vae_enc_io_binding.bind_output(name=vae_enc_output_name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(encoded_latent_8_channel.shape), buffer_ptr=encoded_latent_8_channel.data_ptr())
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        vae_enc_session.run_with_iobinding(vae_enc_io_binding)
        # --- End VAE Encoder Call ---
        # print(f"DEBUG: Denoiser - encoded_latent_8_channel (VAE Enc output, UNscaled) min: {encoded_latent_8_channel.min():.4f}, max: {encoded_latent_8_channel.max():.4f}, mean: {encoded_latent_8_channel.mean():.4f}")

        # Prepare UNet inputs
        lq_latent_scaled_for_unet = encoded_latent_8_channel * self.vae_scale_factor
        # print(f"DEBUG: Denoiser - lq_latent_scaled_for_unet (z_lq_scaled for UNet 'lq' part) min: {lq_latent_scaled_for_unet.min():.4f}, max: {lq_latent_scaled_for_unet.max():.4f}, mean: {lq_latent_scaled_for_unet.mean():.4f}")

        dummy_context = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device).contiguous() # Minimal context
        dummy_class_labels = torch.zeros((1,), dtype=torch.int64, device=self.device).contiguous()   # Minimal class labels
        is_ref_flag = torch.tensor(False, dtype=torch.bool, device=self.device).contiguous()        # Denoising, not reference

        pred_x0_unscaled = torch.empty_like(encoded_latent_8_channel)

        # Always use Single Step Fast mode as DDIM is removed
        # print(f"DEBUG: Denoiser - Mode: Single Step, Timestep t={denoiser_single_step_t}")
        x0_unscaled_for_noise_addition = encoded_latent_8_channel 
        timesteps_tensor = torch.tensor([denoiser_single_step_t], dtype=torch.int64, device=self.device)

        alpha_t_val = self.alphas_cumprod_np[denoiser_single_step_t]
        sqrt_alpha_bar_t = torch.sqrt(torch.tensor(alpha_t_val, device=self.device, dtype=torch.float32))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - torch.tensor(alpha_t_val, device=self.device, dtype=torch.float32))

        # Seed for deterministic noise generation for this frame and timestep
        torch.manual_seed(frame_number_for_seed + denoiser_single_step_t)
        noise_sample = torch.randn_like(x0_unscaled_for_noise_addition)
        xt_noisy_unscaled_8_channel = x0_unscaled_for_noise_addition * sqrt_alpha_bar_t + noise_sample * sqrt_one_minus_alpha_bar_t

        unet_input_16_channel = torch.cat((xt_noisy_unscaled_8_channel, lq_latent_scaled_for_unet), dim=1)
        predicted_noise_from_unet = torch.empty((1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device).contiguous()
        self.run_ref_ldm_unet(unet_filename, unet_input_16_channel, timesteps_tensor, dummy_context, dummy_class_labels, is_ref_flag, predicted_noise_from_unet)
        
        pred_x0_unscaled = (xt_noisy_unscaled_8_channel - sqrt_one_minus_alpha_bar_t * predicted_noise_from_unet) / sqrt_alpha_bar_t

        # Latent masking removed as it might be causing white backgrounds.

        # print(f"DEBUG: Denoiser - pred_x0_unscaled (calculated, to VAE Dec) min: {pred_x0_unscaled.min():.4f}, max: {pred_x0_unscaled.max():.4f}, mean: {pred_x0_unscaled.mean():.4f}")
        # torch.use_deterministic_algorithms(False) # Reset if it was set earlier
        # Reset deterministic algorithms to previous state
        torch.use_deterministic_algorithms(old_deterministic_state)

        # 4. VAE Decode - VAE decoder expects UNSCALED latent based on example code
        latent_for_vae_decoder = pred_x0_unscaled
        decoded_image_normalized_bchw = torch.empty((1, 3, h_proc, w_proc), dtype=torch.float32, device=self.device).contiguous()

        # --- VAE Decoder Call ---
        vae_dec_model_name = 'RefLDMVAEDecoder'
        if not self.models.get(vae_dec_model_name) or self.models[vae_dec_model_name] is None:
            self.models[vae_dec_model_name] = self.load_model(vae_dec_model_name)

        vae_dec_session = self.models[vae_dec_model_name]
        vae_dec_input_name = vae_dec_session.get_inputs()[0].name
        vae_dec_output_name = vae_dec_session.get_outputs()[0].name

        vae_dec_io_binding = vae_dec_session.io_binding()
        vae_dec_io_binding.bind_input(name=vae_dec_input_name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(latent_for_vae_decoder.shape), buffer_ptr=latent_for_vae_decoder.data_ptr())
        vae_dec_io_binding.bind_output(name=vae_dec_output_name, device_type=self.device, device_id=0, element_type=np.float32, shape=tuple(decoded_image_normalized_bchw.shape), buffer_ptr=decoded_image_normalized_bchw.data_ptr())
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device != "cpu":
            self.syncvec.cpu()
        vae_dec_session.run_with_iobinding(vae_dec_io_binding)
        # --- End VAE Decoder Call ---

        # print(f"DEBUG: Denoiser - decoded_image_normalized_bchw (VAE Dec output) min: {decoded_image_normalized_bchw.min():.4f}, max: {decoded_image_normalized_bchw.max():.4f}, mean: {decoded_image_normalized_bchw.mean():.4f}")

        # 5. Denormalize image from [-1, 1] to [0, 255] uint8
        denoised_image_cxhxw_uint8 = ((decoded_image_normalized_bchw.squeeze(0) + 1.0) * 127.5).clamp(0, 255).byte()
        
        # Apply RGB mask to the denoised output to clean up background before returning
        # self.lp_mask_crop is initialized as [1, 512, 512] in ModelsProcessor from faceutil.create_faded_inner_mask
        # It needs to be expanded to 3 channels if it's single channel.
        # This mask should be 1 in the center (face region) and fade to 0 at the borders.

        # Use a clone of lp_mask_crop to avoid modifying the shared instance if it's used elsewhere directly
        # and to ensure it's on the correct device.
        current_device = denoised_image_cxhxw_uint8.device
        # Ensure self.lp_mask_crop is on the correct device before cloning and repeating
        rgb_mask_for_blend = self.lp_mask_crop.to(current_device).clone() 

        if rgb_mask_for_blend.shape[0] == 1: # If it's [1, H, W]
            rgb_mask_for_blend = rgb_mask_for_blend.repeat(3, 1, 1) # Repeat to [3, H, W]

        # Aggressively ensure the mask's borders are zero to prevent edge artifacts.
        # This makes the outer edge of the mask hard zero.
        border_px_for_mask_zeroing = 10  # Number of pixels from edge of the mask to force to zero
        if rgb_mask_for_blend.ndim == 3: # CxHxW
            rgb_mask_for_blend[:, :border_px_for_mask_zeroing, :] = 0  # Top
            rgb_mask_for_blend[:, -border_px_for_mask_zeroing:, :] = 0 # Bottom
            rgb_mask_for_blend[:, :, :border_px_for_mask_zeroing] = 0  # Left
            rgb_mask_for_blend[:, :, -border_px_for_mask_zeroing:] = 0 # Right

        if denoised_image_cxhxw_uint8.shape == rgb_mask_for_blend.shape:
            denoised_float = denoised_image_cxhxw_uint8.float()
            black_background = torch.zeros_like(denoised_float) # Ensure background is black
            masked_denoised_float = denoised_float * rgb_mask_for_blend + black_background * (1.0 - rgb_mask_for_blend)
            denoised_image_cxhxw_uint8 = masked_denoised_float.clamp(0, 255).byte()
        else:
            print(f"Warning: RGB output mask shape {rgb_mask_for_blend.shape} does not match denoised image shape {denoised_image_cxhxw_uint8.shape}. Skipping RGB output mask.")
        return denoised_image_cxhxw_uint8