"""
Unified advanced control for CorePulse.

This module provides a single injector that combines all advanced control types:
- Prompt injection (text conditioning)
- Attention manipulation (cross-attention control)
- Self-attention manipulation (image-image attention)
- Multi-scale control (resolution-aware injection)

All controls are applied in a single pass for maximum efficiency and compatibility.
"""

from typing import Union, Dict, List, Optional, Any
import torch
from diffusers import DiffusionPipeline

from .advanced import AdvancedPromptInjector
from ..models.base import BlockIdentifier
from ..models.unet_patcher import AttentionMapConfig, SelfAttentionConfig
from ..utils.tokenizer import find_token_indices
from ..utils.logger import logger


class UnifiedAdvancedInjector(AdvancedPromptInjector):
    """
    Unified injector that provides all advanced control types in a single pass.
    
    This combines:
    - Prompt injection (different text conditioning per block)
    - Attention manipulation (cross-attention control for specific tokens)
    - Self-attention manipulation (image-to-image attention control)  
    - Multi-scale control (resolution-aware conditioning)
    
    All controls are configured and applied in a single pipeline modification.
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """
        Initialize the unified advanced injector.
        
        Args:
            pipeline: The diffusers pipeline to manipulate
        """
        super().__init__(pipeline)
        
        # Get tokenizers for attention manipulation
        self.tokenizer = getattr(pipeline, 'tokenizer', None)
        self.tokenizer_2 = getattr(pipeline, 'tokenizer_2', None)
        
        if not self.tokenizer:
            logger.warning("Pipeline has no tokenizer - attention manipulation will be disabled")
        
        logger.info(f"Initialized UnifiedAdvancedInjector for {self.model_type} model")

    # ==============================================
    # ATTENTION MANIPULATION (Cross-Attention)
    # ==============================================

    def add_attention_manipulation(self,
                                   prompt: str,
                                   block: Union[str, BlockIdentifier],
                                   target_phrase: str,
                                   attention_scale: float = 1.0,
                                   sigma_start: float = 1.0,
                                   sigma_end: float = 0.0,
                                   spatial_mask: Optional[torch.Tensor] = None):
        """
        Add cross-attention manipulation for specific words/phrases.
        
        Args:
            prompt: The prompt containing the target phrase
            block: Block identifier or "all" 
            target_phrase: Specific phrase to manipulate (e.g., "castle")
            attention_scale: Scaling factor for attention weights
            sigma_start: Start sigma for manipulation window
            sigma_end: End sigma for manipulation window
            spatial_mask: Optional spatial mask for regional control
        """
        if not self.tokenizer:
            logger.error("Cannot add attention manipulation - no tokenizer available")
            raise ValueError("Pipeline must have tokenizer for attention manipulation")
        
        logger.info(f"Adding attention manipulation: '{target_phrase}' in block '{block}' with scale {attention_scale}")
        
        # Find token indices for the target phrase
        all_token_indices = []

        # Find indices with the first tokenizer
        indices_1 = find_token_indices(self.tokenizer, prompt, target_phrase)
        if indices_1:
            all_token_indices.extend(indices_1)

        # Find indices with the second tokenizer if it exists (SDXL)
        if self.tokenizer_2:
            indices_2 = find_token_indices(self.tokenizer_2, prompt, target_phrase)
            if indices_2:
                # Offset the indices by the max length of the first tokenizer's context
                offset = self.tokenizer.model_max_length
                all_token_indices.extend([idx + offset for idx in indices_2])
        
        if not all_token_indices:
            raise ValueError(f"Could not find target phrase '{target_phrase}' in prompt '{prompt}'")

        # Add the manipulation to the patcher
        self.patcher.add_attention_manipulation(
            block=block,
            target_token_indices=all_token_indices,
            attention_scale=attention_scale,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            spatial_mask=spatial_mask
        )

    # ==============================================
    # SELF-ATTENTION MANIPULATION (Image-Image)
    # ==============================================

    def add_self_attention_manipulation(self,
                                      block: Union[str, BlockIdentifier],
                                      source_region: Optional[torch.Tensor] = None,
                                      target_region: Optional[torch.Tensor] = None,
                                      attention_scale: float = 1.0,
                                      interaction_type: str = "enhance",
                                      sigma_start: float = 1.0,
                                      sigma_end: float = 0.0):
        """
        Add self-attention manipulation for image-to-image attention control.
        
        Args:
            block: Block identifier or "all"
            source_region: Spatial mask for pixels whose attention to modify
            target_region: Spatial mask for pixels to attend to
            attention_scale: Scaling factor for attention weights
            interaction_type: "enhance", "suppress", "redirect", "sharpen", "smooth"
            sigma_start: Start sigma for manipulation window
            sigma_end: End sigma for manipulation window
        """
        logger.info(f"Adding self-attention manipulation: {interaction_type} with scale {attention_scale} in block '{block}'")
        
        self.patcher.add_self_attention_manipulation(
            block=block,
            source_region=source_region,
            target_region=target_region,
            attention_scale=attention_scale,
            interaction_type=interaction_type,
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

    def enhance_region_interaction(self,
                                 source_region: torch.Tensor,
                                 target_region: torch.Tensor,
                                 attention_scale: float = 2.0,
                                 block: Union[str, BlockIdentifier] = "all",
                                 sigma_start: float = 1.0,
                                 sigma_end: float = 0.0):
        """Enhance how one image region attends to another."""
        logger.debug(f"Enhancing region interaction: {source_region.sum().item():.0f} -> {target_region.sum().item():.0f} pixels")
        
        self.add_self_attention_manipulation(
            block=block,
            source_region=source_region,
            target_region=target_region,
            attention_scale=attention_scale,
            interaction_type="enhance",
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

    def enhance_global_coherence(self,
                               attention_scale: float = 1.5,
                               block: Union[str, BlockIdentifier] = "middle:0",
                               sigma_start: float = 1.0,
                               sigma_end: float = 0.5):
        """Globally enhance self-attention to improve image coherence."""
        logger.info(f"Adding global coherence enhancement with scale {attention_scale}")
        
        self.add_self_attention_manipulation(
            block=block,
            source_region=None,  # Global application
            target_region=None,
            attention_scale=attention_scale,
            interaction_type="enhance",
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )

    # ==============================================
    # MULTI-SCALE CONTROL (Resolution-Aware)
    # ==============================================

    def add_detail_injection(self,
                           prompt: str,
                           weight: float = 2.2,  # Balanced default weight
                           resolution_levels: List[str] = ["highest", "high"],
                           sigma_start: float = 0.7,  # Target detail refinement phase
                           sigma_end: float = 0.0,    # Through final details
                           spatial_mask: Optional[torch.Tensor] = None):
        """
        Inject prompts that control fine details and textures.
        """
        logger.info(f"Adding detail injection: '{prompt}' at {resolution_levels} resolution")
        
        target_blocks = []
        for resolution in resolution_levels:
            blocks = self.patcher.block_mapper.get_blocks_by_resolution(resolution)
            target_blocks.extend(blocks)
        
        if not target_blocks:
            logger.warning(f"No blocks found for resolution levels: {resolution_levels}")
            return
        
        for block in target_blocks:
            self.add_injection(
                block=block,
                prompt=prompt,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                spatial_mask=spatial_mask
            )
        
        logger.debug(f"Applied detail injection to {len(target_blocks)} blocks")

    def add_structure_injection(self,
                              prompt: str,
                              weight: float = 2.5,  # Balanced structural weight
                              resolution_levels: List[str] = ["lowest"],  # Focus on bottleneck
                              sigma_start: float = 1.0,  # Target composition phase
                              sigma_end: float = 0.3,    # Through mid-generation
                              spatial_mask: Optional[torch.Tensor] = None):
        """
        Inject prompts that control overall structure and composition.
        """
        logger.info(f"Adding structure injection: '{prompt}' at {resolution_levels} resolution")
        
        target_blocks = []
        for resolution in resolution_levels:
            blocks = self.patcher.block_mapper.get_blocks_by_resolution(resolution)
            target_blocks.extend(blocks)
        
        if not target_blocks:
            logger.warning(f"No blocks found for resolution levels: {resolution_levels}")
            return
        
        for block in target_blocks:
            self.add_injection(
                block=block,
                prompt=prompt,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                spatial_mask=spatial_mask
            )
        
        logger.debug(f"Applied structure injection to {len(target_blocks)} blocks")

    def add_hierarchical_prompts(self,
                               structure_prompt: str,
                               detail_prompt: str,
                               midlevel_prompt: Optional[str] = None,
                               weights: Optional[Dict[str, float]] = None,
                               sigma_start: float = 0.0,
                               sigma_end: float = 1.0):
        """
        Add a complete hierarchical prompt system with different conditioning at each scale.
        """
        logger.info("Adding hierarchical multi-scale prompt system")
        
        # Balanced default weights for noticeable but controlled multi-scale effects
        default_weights = {"structure": 3.0, "midlevel": 2.5, "detail": 2.2}
        weights = {**default_weights, **(weights or {})}
        
        # Structure (low resolution)
        self.add_structure_injection(
            structure_prompt,
            weight=weights.get("structure", 3.0),
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )
        
        # Mid-level (medium resolution)
        if midlevel_prompt:
            target_blocks = []
            for resolution in ["medium", "high"]:
                blocks = self.patcher.block_mapper.get_blocks_by_resolution(resolution)
                target_blocks.extend(blocks)
            
            for block in target_blocks:
                self.add_injection(
                    block=block,
                    prompt=midlevel_prompt,
                    weight=weights.get("midlevel", 2.5),
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
        
        # Details (high resolution)
        self.add_detail_injection(
            detail_prompt,
            weight=weights.get("detail", 2.2),
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )
        
        logger.info("Hierarchical prompt system configured")

    # ==============================================
    # UNIFIED CONTROL METHODS
    # ==============================================

    def add_complete_control(self,
                           base_prompt: str,
                           # Multi-scale prompts
                           structure_prompt: Optional[str] = None,
                           detail_prompt: Optional[str] = None,
                           midlevel_prompt: Optional[str] = None,
                           # Attention manipulations
                           attention_targets: Optional[List[Dict]] = None,
                           # Self-attention manipulations  
                           self_attention_configs: Optional[List[Dict]] = None,
                           # Global settings
                           weights: Optional[Dict[str, float]] = None,
                           sigma_start: float = 0.0,
                           sigma_end: float = 1.0):
        """
        Add complete unified control in a single method call.
        
        Args:
            base_prompt: Base prompt for the generation
            structure_prompt: Low-resolution structural conditioning
            detail_prompt: High-resolution detail conditioning
            midlevel_prompt: Medium-resolution feature conditioning
            attention_targets: List of attention manipulation configs
            self_attention_configs: List of self-attention manipulation configs
            weights: Weight mapping for different control types
            sigma_start: Global start sigma
            sigma_end: Global end sigma
            
        Example:
            injector.add_complete_control(
                base_prompt="a gothic cathedral",
                structure_prompt="majestic cathedral silhouette",
                detail_prompt="intricate stone carvings",
                attention_targets=[
                    {"target_phrase": "gothic", "attention_scale": 2.0, "block": "all"}
                ],
                self_attention_configs=[
                    {"interaction_type": "enhance", "attention_scale": 1.8, "block": "middle:0"}
                ]
            )
        """
        logger.info("Configuring complete unified control system")
        
        weights = weights or {}
        
        # 1. Add hierarchical prompts if provided
        if structure_prompt or detail_prompt:
            self.add_hierarchical_prompts(
                structure_prompt=structure_prompt or base_prompt,
                detail_prompt=detail_prompt or base_prompt,
                midlevel_prompt=midlevel_prompt,
                weights=weights,
                sigma_start=sigma_start,
                sigma_end=sigma_end
            )
        
        # 2. Add attention manipulations
        if attention_targets:
            for config in attention_targets:
                self.add_attention_manipulation(
                    prompt=config.get("prompt", base_prompt),
                    block=config.get("block", "all"),
                    target_phrase=config["target_phrase"],
                    attention_scale=config.get("attention_scale", 1.0),
                    sigma_start=config.get("sigma_start", sigma_start),
                    sigma_end=config.get("sigma_end", sigma_end),
                    spatial_mask=config.get("spatial_mask")
                )
        
        # 3. Add self-attention manipulations
        if self_attention_configs:
            for config in self_attention_configs:
                self.add_self_attention_manipulation(
                    block=config.get("block", "all"),
                    source_region=config.get("source_region"),
                    target_region=config.get("target_region"),
                    attention_scale=config.get("attention_scale", 1.0),
                    interaction_type=config.get("interaction_type", "enhance"),
                    sigma_start=config.get("sigma_start", sigma_start),
                    sigma_end=config.get("sigma_end", sigma_end)
                )
        
        logger.info("Complete unified control system configured - ready for single-pass application")

    def get_control_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all configured controls.
        """
        return {
            "prompt_injections": len(self.configs),
            "attention_manipulations": len(self.patcher.attention_map_configs),
            "self_attention_manipulations": len(self.patcher.self_attention_configs),
            "total_configurations": (
                len(self.configs) + 
                len(self.patcher.attention_map_configs) + 
                len(self.patcher.self_attention_configs)
            )
        }
