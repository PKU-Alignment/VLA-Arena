import torch
import os
import sys
import numpy as np
import json
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# Add the openvla path
sys.path.append('/DATA/disk0/borong/openvla')

from vla_arena.evaluation.policy.prismatic_for_openvla import *
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry 
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action, read_eval_cfgs
from vla_arena.evaluation.openvla_utils import (
    resize_image_for_policy,
    center_crop_image,
)

# Import LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def copy_file_content(content_file, target_file):
    """Copy content from one file to another."""
    with open(content_file, "r") as f:
        content = f.read()
    with open(target_file, "w") as f:
        f.write(content)


@PolicyRegistry.register("openvla")
class OpenVLA(Policy):
    """OpenVLA Policy for robot action prediction."""
    
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    
    def __init__(self, 
                 model_ckpt,
                 eval_cfgs_path='../../configs/evaluation/openvla.yaml',
                 attn_implementation=None,
                 norm_config_file=None,
                 device="cuda",
                 **kwargs):
        """
        Initialize OpenVLA policy.
        
        Args:
            model_ckpt: Path to the model checkpoint
            attn_implementation: The implementation of attention layer (e.g., "torch" or "einsum")
            norm_config_file: Path to the config file for denormalization to override the default config
            device: Device to run on ("cuda" or "cpu")
            **kwargs: Additional arguments including 'instruction'
        """
        eval_cfgs = read_eval_cfgs(self.name, eval_cfgs_path)
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Override config if norm_config_file is provided
        if norm_config_file is not None:
            copy_file_content(norm_config_file, os.path.join(model_ckpt, "config.json"))
        
        # Add model directory to Python path
        if model_ckpt not in sys.path:
            sys.path.insert(0, model_ckpt)
            print(f"Added {model_ckpt} to Python path")
        
        # Load model components
        print("Loading OpenVLA model...")
        with open(os.path.join(model_ckpt, "dataset_statistics.json"), "r") as f:
            norm_stats = json.load(f)
        # Load configuration
        config = OpenVLAConfig.from_pretrained(
            model_ckpt,
            local_files_only=True,
            trust_remote_code=True,
            norm_stats=norm_stats
        )
        
        # Load processor
        self.processor = PrismaticProcessor.from_pretrained(
            model_ckpt,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        # Load model
        model = OpenVLAForActionPrediction.from_pretrained(
            model_ckpt,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        print("Model loaded successfully!")
        
        # Move model to the specified device
        model = model.to(device)
        print(f"Model moved to device: {device}")
        
        # Store instruction if provided
        self.instruction = kwargs.get('instruction', None)
        self.device = device
        self.center_crop = eval_cfgs.get("center_crop", True)
        # Call parent class constructor
        super().__init__(model)
    
    def _process_observation(self, obs, unnorm_key=None):
        """Prepare inputs for the model."""
        prompt = self._build_prompt()
        img = obs["agentview_image"]
        # resize image to 224x224
        img = resize_image_for_policy(img, 224)
        # Flip image if needed
        img = img[::-1, ::-1]
        # center crop image
        if self.center_crop:
            img = center_crop_image(img)
        inputs = self.processor(
            prompt, 
            Image.fromarray(img).convert("RGB")
        ).to(self.device, dtype=torch.bfloat16)
        return inputs
    
    def _build_prompt(self):
        """Build the prompt for the model."""
        prompt = f"In: What action should the robot take to {self.instruction}?\nOut: "
        return prompt
    
    def predict(self, obs, unnorm_key=None):
        """Predict action given observation."""
        inputs = self._prepare_observation(obs, unnorm_key)
        action = self.model.predict_action(
            **inputs, 
            do_sample=False, 
            unnorm_key=unnorm_key
        )
        action = self._process_action(action)
        return action
    
    def _process_action(self, action):
        """Process the predicted action."""
        action = normalize_gripper_action(action)
        action = invert_gripper_action(action)
        return action
    
    @property
    def name(self):
        """Return the name of the policy."""
        return "OpenVLA"