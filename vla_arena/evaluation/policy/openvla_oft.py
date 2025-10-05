import torch
import os
import sys
import numpy as np
import json
import peft
import imageio
from collections import deque
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action, read_eval_cfgs
from vla_arena.evaluation.policy.prismatic_for_openvla_oft import *
from vla_arena.evaluation.openvla_utils import (
    resize_image_for_policy,
    find_checkpoint_file,
    load_component_state_dict,
    prepare_images_for_vla,
    normalize_proprio,
)


def copy_file_content(content_file, target_file):
    with open(content_file, "r") as f:
        content = f.read()
    with open(target_file, "w") as f:
        f.write(content)


def quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    
    # Extract scalar and vector parts
    w, x, y, z = quat
    
    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    # Compute axis
    sin_half_angle = np.sqrt(1 - w * w)
    if sin_half_angle < 1e-10:
        # If angle is close to 0, axis doesn't matter
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([x, y, z]) / sin_half_angle
    
    return axis * angle

@PolicyRegistry.register("openvla-oft")
class OpenVLAOFT(Policy):
    """OpenVLA with Online Fine-Tuning capabilities."""
    
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    
    def __init__(self, 
                 model_ckpt,
                 attn_implementation=None,
                 norm_config_file=None,
                 device="cuda",
                 unnorm_key=None,
                 # OFT specific parameters
                 eval_cfgs_path='../../configs/evaluation/openvla_oft.yaml',
                 # Logging parameters
                 enable_input_logging=False,
                 logging_dir="./model_input_logs",
                 **kwargs):
        """
        Initialize OpenVLA with Online Fine-Tuning capabilities.
        
        Args:
            model_ckpt: Path to the model checkpoint
            attn_implementation: The implementation of attention layer (e.g., "torch" or "einsum")
            norm_config_file: Path to the config file for denormalization to override the default config
            device: Device to run on ("cuda" or "cpu")
            
            OFT specific parameters:
            use_l1_regression: If True, uses continuous action head with L1 regression objective
            use_diffusion: If True, uses continuous action head with diffusion modeling objective
            num_diffusion_steps_train: Number of diffusion steps used for training
            num_diffusion_steps_inference: Number of diffusion steps used for inference
            use_film: If True, uses FiLM to infuse language inputs into visual features
            num_images_in_input: Number of images in the VLA input
            use_proprio: Whether to include proprio state in input
            center_crop: Center crop? (if trained w/ random crop image aug)
            num_open_loop_steps: Number of actions to execute open-loop before requerying policy
            lora_rank: Rank of LoRA weight matrix
            load_in_8bit: Load with 8-bit quantization
            load_in_4bit: Load with 4-bit quantization
            enable_input_logging: Whether to log all model inputs
            logging_dir: Directory to save input logs
            **kwargs: Additional arguments including 'instruction'
        """
        
        # Read evaluation configs and store OFT parameters
        eval_cfgs = read_eval_cfgs(self.name, eval_cfgs_path)
        self.unnorm_key = eval_cfgs.get("unnorm_key", unnorm_key)
        self.unnorm_key = "libero_spatial_no_noops" if self.unnorm_key is None else self.unnorm_key
        self.use_l1_regression = eval_cfgs.get("use_l1_regression", True)
        self.use_diffusion = eval_cfgs.get("use_diffusion", False)
        self.num_diffusion_steps_train = eval_cfgs.get("num_diffusion_steps_train", 50)
        self.num_diffusion_steps_inference = eval_cfgs.get("num_diffusion_steps_inference", 50)
        self.use_film = eval_cfgs.get("use_film", True)
        self.num_images_in_input = eval_cfgs.get("num_images_in_input", 2)
        self.use_proprio = eval_cfgs.get("use_proprio", False)
        self.center_crop = eval_cfgs.get("center_crop", True)
        self.image_resize_size = eval_cfgs.get("image_resize_size", 224)
        self.lora_rank = eval_cfgs.get("lora_rank", 32)
        self.llm_dim = eval_cfgs.get("llm_dim", 4096)
        self.load_in_8bit = eval_cfgs.get("load_in_8bit", False)
        self.load_in_4bit = eval_cfgs.get("load_in_4bit", False)
        self.num_open_loop_steps = eval_cfgs.get("num_open_loop_steps", 8)
        self.device = device
        
        # Initialize logging parameters
        self.enable_input_logging = enable_input_logging
        self.logging_dir = logging_dir
        self.input_logs = {
            'agentview_images': [],
            'wrist_images': [],
            'prompts': [],
            'proprio_data': [],
            'actions': [],
            'timestamps': [],
            'input_tensors': [],
            'pixel_values': [],
            'input_ids': [],
            'attention_mask': []
        }
        
        # Create logging directory if enabled
        if self.enable_input_logging:
            self.logging_dir = Path(self.logging_dir)
            self.logging_dir.mkdir(parents=True, exist_ok=True)
            print(f"Input logging enabled. Logs will be saved to: {self.logging_dir}")

        # Initialize action queue for open-loop control
        self.action_queue = deque(maxlen=self.num_open_loop_steps)
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Override config if norm_config_file is provided
        if norm_config_file is not None:
            copy_file_content(norm_config_file, os.path.join(model_ckpt, "config.json"))
        
        # Add model directory to Python path
        if model_ckpt not in sys.path:
            sys.path.insert(0, model_ckpt)
            print(f"Added {model_ckpt} to Python path")
        
        # Load model components
        print("Loading OpenVLA-OFT model...")
        with open(os.path.join(model_ckpt, "dataset_statistics.json"), "r") as f:
            norm_stats = json.load(f)
        # Load configuration
        config = OpenVLAConfig.from_pretrained(
            model_ckpt,
            local_files_only=True,
            trust_remote_code=True,
            norm_stats=norm_stats,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit
        )
        print('config loaded successfully!')
        # Load processor
        self.processor = PrismaticProcessor.from_pretrained(
            model_ckpt,
            local_files_only=True,
            trust_remote_code=True
        )
        print("Processor loaded successfully!")
        # Prepare model loading kwargs based on quantization options
        model_kwargs = {
            "config": config,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "trust_remote_code": True
        }
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        # Load model
        self.model = OpenVLAForActionPrediction.from_pretrained(
            model_ckpt,
            **model_kwargs
        )
        
        print("Model loaded successfully!")
        
        # Move model to device if not quantized
        if not (self.load_in_8bit or self.load_in_4bit):
            self.model = self.model.to(device)
        print(f"Model moved to device: {device}")
        
        # Store model checkpoint path for later use
        self.model_ckpt = model_ckpt
        
        # Initialize additional components for OFT
        self._initialize_oft_components()
        
        # Store instruction if provided
        self.instruction = kwargs.get('instruction', None)
        self.device = device
        
        # Call parent class constructor
        super().__init__(self.model)
    
    def _initialize_oft_components(self):
        """Initialize Online Fine-Tuning specific components"""
        # Create a config object that the imported functions expect
        
        
        
        # Initialize proprio projector using imported function
        self.proprio_projector = None
        if self.use_proprio:
            self.proprio_projector = self._get_proprio_projector()
        
        # Initialize action head using imported function
        self.action_head = None
        if self.use_l1_regression or self.use_diffusion:
            self.action_head = self._get_action_head(self.llm_dim)
        
        # Initialize noisy action projector using imported function
        self.noisy_action_projector = None
        if self.use_diffusion:
            self.noisy_action_projector = self._get_noisy_action_projector(self.llm_dim)
        
        if self.use_film:
            # Apply FiLM to the model using imported function
            self.model = self._apply_film_to_vla()

        # Check and set unnorm key for action normalization
        self._check_unnorm_key()
    def _get_noisy_action_projector(self, llm_dim: int) -> NoisyActionProjector:
        """
        Get noisy action projector for diffusion-based action prediction.

        Args:      
            llm_dim: Dimension of the language model

        Returns:
            NoisyActionProjector: The initialized noisy action projector
        """
        # Initialize projector and move to device
        noisy_action_projector = NoisyActionProjector(
            llm_dim=llm_dim,
        ).to(self.device)
        noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(self.device)
        noisy_action_projector.eval()

        # Find and load checkpoint
        checkpoint_path = find_checkpoint_file(self.model_ckpt, "noisy_action_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        noisy_action_projector.load_state_dict(state_dict)

        return noisy_action_projector

    def _check_unnorm_key(self):
        """Check and set the action unnormalization key"""        
        # Check if model has norm_stats
        if hasattr(self.model, 'norm_stats'):
            # Try to find the appropriate unnorm key
            possible_keys = [
                "libero_spatial",
                "libero_object", 
                "libero_goal",
                "libero_10",
                "libero_90",
                "libero_spatial_no_noops",
                "libero_object_no_noops",
                "libero_goal_no_noops",
                "libero_10_no_noops",
                "libero_90_no_noops",
                "vla_arena"
            ]
            
            for key in possible_keys:
                if key in self.model.norm_stats:
                    self.unnorm_key = key
                    print(f"Found unnorm key: {self.unnorm_key}")
                    break
            
            if self.unnorm_key is None:
                print("Warning: No unnorm key found in model.norm_stats")
                print(f"Available keys: {list(self.model.norm_stats.keys()) if hasattr(self.model, 'norm_stats') else 'None'}")
        else:
            print("Warning: Model does not have norm_stats attribute")
    
    def _get_proprio_projector(self, llm_dim=4096, proprio_dim=8) -> ProprioProjector:
        """
        Get proprioception projector for the VLA model.

        Args:  
            llm_dim: Dimension of the language model
            proprio_dim: Dimension of proprioception data

        Returns:
            ProprioProjector: The initialized proprio projector
        """
        # Initialize projector and move to device
        proprio_projector = ProprioProjector(
            llm_dim=llm_dim,
            proprio_dim=proprio_dim,
        ).to(self.device)
        proprio_projector = proprio_projector.to(torch.bfloat16).to(self.device)
        proprio_projector.eval()

        checkpoint_path = find_checkpoint_file(self.model_ckpt, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

        return proprio_projector
    
    def _get_action_head(self, llm_dim=4096) -> Union[L1RegressionActionHead, DiffusionActionHead]:
        """
        Get action head for continuous value prediction.

        Args:
            llm_dim: Dimension of the language model

        Returns:
            Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

        Raises:
            AssertionError: If both L1 regression and diffusion are specified
        """
        assert not (self.use_l1_regression and self.use_diffusion), "Cannot use both L1 regression and diffusion action head!"

        # Initialize appropriate action head based on configuration
        if self.use_l1_regression:             
            action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)         
        # elif self.use_diffusion:             
        #     action_head = DiffusionActionHead(
        #         input_dim=llm_dim, 
        #         hidden_dim=llm_dim, 
        #         action_dim=ACTION_DIM,                                   num_diffusion_steps_train=              self.num_diffusion_steps_train)
        # 
        #                                                   action_head.noise_scheduler.set_timesteps(self.num_diffusion_steps_inference)   
        #         
        else:             
            raise ValueError("Either use_l1_regression or use_diffusion must be True")

        action_head = action_head.to(torch.bfloat16).to(self.device)
        action_head.eval()
        checkpoint_path = find_checkpoint_file(self.model_ckpt, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

        return action_head

    
    def _apply_film_to_vla(self) -> torch.nn.Module:
        """
        Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.

        Args:
            vla: The VLA model

        Returns:
            torch.nn.Module: VLA model with FiLM applied
        """
        from peft import LoraConfig, get_peft_model

        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=min(self.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(self.model, lora_config)

        # Create and apply FiLMed vision backbone
        new_vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
        )
        vla.model.vision_backbone = new_vision_backbone

        # Load vision backbone checkpoint
        checkpoint_path = find_checkpoint_file(self.model_ckpt, "vision_backbone")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        vla.model.vision_backbone.load_state_dict(state_dict)

        # Use the model component instead of wrapper and convert to bfloat16
        vla = vla.model
        vla.vision_backbone = vla.vision_backbone.to(self.device, dtype=torch.bfloat16)

        return vla.to(self.device)
    
    def _save_input_logs(self):
        """Save all logged inputs to files and videos"""
        if not self.enable_input_logging or not any(self.input_logs.values()):
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = self.logging_dir / f"episode_{timestamp}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Save agentview images as video
        if self.input_logs['agentview_images']:
            agentview_video_path = episode_dir / "agentview_images.mp4"
            video_writer = imageio.get_writer(agentview_video_path, fps=30)
            for img in self.input_logs['agentview_images']:
                video_writer.append_data(img)
            video_writer.close()
            print(f"Saved agentview video: {agentview_video_path}")
        
        # Save wrist images as video
        if self.input_logs['wrist_images']:
            wrist_video_path = episode_dir / "wrist_images.mp4"
            video_writer = imageio.get_writer(wrist_video_path, fps=30)
            for img in self.input_logs['wrist_images']:
                video_writer.append_data(img)
            video_writer.close()
            print(f"Saved wrist video: {wrist_video_path}")
        
        # Save other data as JSON
        other_data = {
            'prompts': self.input_logs['prompts'],
            'proprio_data': self.input_logs['proprio_data'],
            'actions': self.input_logs['actions'],
            'timestamps': self.input_logs['timestamps'],
            'input_tensors': self.input_logs['input_tensors'],
            'pixel_values': self.input_logs['pixel_values'],
            'input_ids': self.input_logs['input_ids'],
            'attention_mask': self.input_logs['attention_mask'],
            'instruction': self.instruction,
            'model_config': {
                'use_l1_regression': self.use_l1_regression,
                'use_diffusion': self.use_diffusion,
                'use_film': self.use_film,
                'num_images_in_input': self.num_images_in_input,
                'use_proprio': self.use_proprio,
                'num_open_loop_steps': self.num_open_loop_steps,
                'unnorm_key': self.unnorm_key
            }
        }
        
        data_file_path = episode_dir / "input_data.json"
        with open(data_file_path, 'w') as f:
            json.dump(other_data, f, indent=2, default=str)
        print(f"Saved input data: {data_file_path}")
        
        # Save individual images as well
        if self.input_logs['agentview_images']:
            images_dir = episode_dir / "agentview_frames"
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(self.input_logs['agentview_images']):
                imageio.imwrite(images_dir / f"frame_{i:04d}.png", img)
        
        if self.input_logs['wrist_images']:
            images_dir = episode_dir / "wrist_frames"
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(self.input_logs['wrist_images']):
                imageio.imwrite(images_dir / f"frame_{i:04d}.png", img)

    def reset_instruction(self, instruction):
        """Reset the policy with a new instruction"""
        self.instruction = instruction
        self.action_queue.clear()
        
        # Save logs if logging is enabled and we have data
        if self.enable_input_logging and any(self.input_logs.values()):
            self._save_input_logs()
        
        # Clear logs for new episode
        self.input_logs = {
            'agentview_images': [],
            'wrist_images': [],
            'prompts': [],
            'proprio_data': [],
            'actions': [],
            'timestamps': [],
            'input_tensors': [],
            'pixel_values': [],
            'input_ids': [],
            'attention_mask': []
        }
    
    def _process_observation(self, obs, unnorm_key=None):
        """Prepare inputs for the model with OFT enhancements"""
        prompt = self._build_prompt(self.instruction)
        
        # Handle multiple image inputs if configured
        if self.num_images_in_input > 1:
            # For multi-image input, we need to handle both current and previous frames
            # This is a simplified version - actual implementation would need frame history
            img = obs.get("agentview_image", obs.get("full_image"))
            wrist_img = obs.get("robot0_eye_in_hand_image", None)
            assert wrist_img is not None, "Wrist image required for multi-image input"
        else:
            img = obs.get("agentview_image", obs.get("full_image"))
            wrist_img = None
        
        all_images = [resize_image_for_policy(img, self.image_resize_size)]
        if wrist_img is not None:
            all_images.append(wrist_img)

        for image in all_images:
            image = image[::-1, ::-1]

        all_images = prepare_images_for_vla(all_images, self.center_crop)
        primary_image = all_images.pop(0)



        # Log input data if enabled
        if self.enable_input_logging:
            timestamp = datetime.now().isoformat()
            self.input_logs['timestamps'].append(timestamp)
            self.input_logs['prompts'].append(prompt)
            
            # Log images (before flipping)
            if img is not None:
                self.input_logs['agentview_images'].append(img.copy())
            if wrist_img is not None:
                self.input_logs['wrist_images'].append(wrist_img.copy())
            
            # Log proprioception data if available
            if self.use_proprio and "state" in obs:
                self.input_logs['proprio_data'].append(obs["state"].tolist() if isinstance(obs["state"], np.ndarray) else obs["state"])

        inputs = self.processor(prompt, Image.fromarray(img).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        
        # Process with OpenVLA processor
        if self.num_images_in_input > 1:
            all_wrist_inputs = [self.processor(prompt, Image.fromarray(wrist_img)).to(self.device, dtype=torch.bfloat16)]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + all_wrist_pixel_values, dim=1)
        
        # Log final input tensors if enabled
        if self.enable_input_logging:
            # Convert tensors to CPU and numpy for serialization
            tensor_data = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    # Convert BFloat16 to float32 before numpy conversion
                    if value.dtype == torch.bfloat16:
                        tensor_data[key] = value.cpu().float().numpy().tolist()
                    else:
                        tensor_data[key] = value.cpu().numpy().tolist()
                else:
                    tensor_data[key] = value
            
            self.input_logs['input_tensors'].append(tensor_data)
            
            # Handle pixel_values with BFloat16 conversion
            if inputs["pixel_values"].dtype == torch.bfloat16:
                self.input_logs['pixel_values'].append(inputs["pixel_values"].cpu().float().numpy().tolist())
            else:
                self.input_logs['pixel_values'].append(inputs["pixel_values"].cpu().numpy().tolist())
            
            if "input_ids" in inputs:
                if inputs["input_ids"].dtype == torch.bfloat16:
                    self.input_logs['input_ids'].append(inputs["input_ids"].cpu().float().numpy().tolist())
                else:
                    self.input_logs['input_ids'].append(inputs["input_ids"].cpu().numpy().tolist())
            
            if "attention_mask" in inputs:
                if inputs["attention_mask"].dtype == torch.bfloat16:
                    self.input_logs['attention_mask'].append(inputs["attention_mask"].cpu().float().numpy().tolist())
                else:
                    self.input_logs['attention_mask'].append(inputs["attention_mask"].cpu().numpy().tolist())
            
        return inputs.to(self.device, dtype=torch.bfloat16)
    
    def _build_prompt(self, instruction):
        """Build prompt for the model"""
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut: "
        return prompt
    
    def predict(self, obs, unnorm_key=None):
        """Predict action with OFT capabilities including open-loop control"""
        # Check if we need to requery the model or use cached actions
        if len(self.action_queue) == 0:
            with torch.inference_mode():
                # Create a config object for get_action function
                inputs = self._process_observation(obs, self.image_resize_size)
                        # Process proprioception data if used
                proprio = None
                if self.use_proprio:
                    proprio = obs["state"]
                    proprio_norm_stats = self.model.norm_stats[self.unnorm_key]["proprio"]
                    obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
                    proprio = obs["state"]
                    proprio = torch.tensor(proprio, dtype=torch.float32, device=self.device)
                if self.action_head is None:
                # Standard VLA output (single-image inputs, discrete actions)
                    actions, _ = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
                else:
                    # Custom action head for continuous actions
                    actions, _ = self.model.predict_action(
                        **inputs,
                        unnorm_key=self.unnorm_key,
                        do_sample=False,
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                        noisy_action_projector=self.noisy_action_projector,
                        action_head=self.action_head,
                        use_film=self.use_film,
                    )
            
            # Add all actions to queue
            if isinstance(actions, list):
                self.action_queue.extend(actions)
            else:
                for action in actions:
                    self.action_queue.append(action)
                    
            # Log actions if enabled
            if self.enable_input_logging:
                if isinstance(actions, list):
                    self.input_logs['actions'].extend([action.tolist() if isinstance(action, np.ndarray) else action for action in actions])
                else:
                    self.input_logs['actions'].append(actions.tolist() if isinstance(actions, np.ndarray) else actions)
                    
        # Get next action from queue
        action = self.action_queue.popleft()
        
        # Process action
        action = self._process_action(action)
        
        return action
    
    def _process_action(self, action):
        """Process action with OFT-specific transformations"""
        # Ensure action is numpy array
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        # Normalize gripper action
        action = normalize_gripper_action(action, binarize=True)
        
        # Invert gripper action for OpenVLA
        action = invert_gripper_action(action)
        
        return action
    
    @property
    def name(self):
        return "OpenVLA-OFT"
    
    @property
    def supports_chunking(self):
        """Whether this policy supports action chunking"""
        return self.num_open_loop_steps > 1
    
    @property
    def chunk_size(self):
        """Size of action chunks"""
        return self.num_open_loop_steps
    
    def save_current_logs(self):
        """Manually save current logs (useful for end of evaluation)"""
        if self.enable_input_logging and any(self.input_logs.values()):
            self._save_input_logs()
    
    def enable_logging(self, logging_dir="./model_input_logs"):
        """Enable input logging with specified directory"""
        self.enable_input_logging = True
        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        print(f"Input logging enabled. Logs will be saved to: {self.logging_dir}")
    
    def disable_logging(self):
        """Disable input logging"""
        if self.enable_input_logging and any(self.input_logs.values()):
            self._save_input_logs()
        self.enable_input_logging = False
        print("Input logging disabled")