"""
Multi-scale control for CorePulse.

This module provides resolution-aware prompt injection and attention control,
allowing different conditioning at different resolution levels of the UNet.
This enables separation of fine details from overall structure and composition.
"""

from typing import Union, Dict, List, Optional, Any
import torch
from diffusers import DiffusionPipeline

from .advanced import AdvancedPromptInjector
from ..models.base import BlockIdentifier
from ..utils.logger import logger


class MultiScaleInjector(AdvancedPromptInjector):
    """
    An injector that provides multi-scale, resolution-aware prompt injection.
    
    This allows different prompts and attention controls at different resolution levels:
    - High resolution: Fine details, textures, small objects
    - Medium resolution: Medium-scale features, faces, local structure  
    - Low resolution: Overall composition, large objects, global structure
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """
        Initialize the multi-scale injector.
        
        Args:
            pipeline: The diffusers pipeline to manipulate
        """
        super().__init__(pipeline)
        logger.info(f"Initialized MultiScaleInjector for {self.model_type} model")
        
        # Pre-compute and cache block mappings for performance
        self._resolution_blocks_cache = self._build_resolution_cache()
        self._stage_blocks_cache = self._build_stage_cache()
        
        logger.debug(f"Cached mappings: {len(self._resolution_blocks_cache)} resolution levels, "
                    f"{len(self._stage_blocks_cache)} stages")

    def _build_resolution_cache(self) -> Dict[str, List[BlockIdentifier]]:
        """Build and cache resolution-to-blocks mapping with pre-created BlockIdentifier objects."""
        cache = {}
        for resolution in ["highest", "high", "medium", "low", "lowest"]:
            block_strings = self.patcher.block_mapper.get_blocks_by_resolution(resolution)
            if block_strings:
                # Pre-create BlockIdentifier objects to avoid string parsing overhead
                cache[resolution] = [BlockIdentifier.from_string(block_str) for block_str in block_strings]
        return cache
    
    def _build_stage_cache(self) -> Dict[str, List[BlockIdentifier]]:
        """Build and cache stage-to-blocks mapping with pre-created BlockIdentifier objects."""
        cache = {}
        for stage in ["downsample", "bottleneck", "upsample"]:
            block_strings = self.patcher.block_mapper.get_blocks_by_stage(stage)
            if block_strings:
                # Pre-create BlockIdentifier objects to avoid string parsing overhead
                cache[stage] = [BlockIdentifier.from_string(block_str) for block_str in block_strings]
        return cache
    
    def _add_prompt_to_blocks(self, 
                             prompt: str,
                             blocks: List[BlockIdentifier],
                             weight: float = 1.0,
                             sigma_start: float = 0.0,
                             sigma_end: float = 1.0,
                             spatial_mask: Optional[torch.Tensor] = None):
        """Efficiently add the same prompt to multiple blocks using pre-created BlockIdentifier objects."""
        if not blocks:
            return
            
        logger.debug(f"Adding prompt to {len(blocks)} blocks: {[str(block) for block in blocks]}")
        for block in blocks:
            # Pass BlockIdentifier directly - no string parsing needed!
            self.add_injection(
                block=block,
                prompt=prompt,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                spatial_mask=spatial_mask
            )

    def add_detail_injection(self,
                           prompt: str,
                           weight: float = 2.2,  # Balanced default weight - noticeable but not overwhelming
                           resolution_levels: List[str] = ["highest", "high"],
                           sigma_start: float = 3.0,  # Wider range - ensures injection on multiple steps
                           sigma_end: float = 0.0,    # Through final details
                           spatial_mask: Optional[torch.Tensor] = None):
        """
        Inject prompts that control fine details and textures.
        
        Args:
            prompt: Prompt for fine details (e.g., "intricate textures, fine details")
            weight: Injection weight
            resolution_levels: Which resolution levels to target
            sigma_start: Start of injection window
            sigma_end: End of injection window
            spatial_mask: Optional spatial mask
            
        Example:
            injector.add_detail_injection(
                "intricate stone textures, weathered surfaces, fine cracks",
                weight=1.2,
                resolution_levels=["highest"]
            )
        """
        logger.debug(f"Adding detail injection: '{prompt}' at {resolution_levels} resolution")
        
        # Use cached mappings for better performance
        target_blocks = []
        for resolution in resolution_levels:
            blocks = self._resolution_blocks_cache.get(resolution, [])
            target_blocks.extend(blocks)
        
        if not target_blocks:
            logger.warning(f"No blocks found for resolution levels: {resolution_levels}")
            return
        
        # Batch operation
        self._add_prompt_to_blocks(prompt, target_blocks, weight, sigma_start, sigma_end, spatial_mask)
        logger.debug(f"Applied detail injection to {len(target_blocks)} blocks")

    def add_structure_injection(self,
                              prompt: str,
                              weight: float = 2.5,  # Balanced structural weight - good influence without overpowering
                              resolution_levels: List[str] = ["lowest"],  # Focus on bottleneck for composition
                              sigma_start: float = 15.0, # Wide range - ensures structural influence
                              sigma_end: float = 0.5,    # Through mid-generation
                              spatial_mask: Optional[torch.Tensor] = None):
        """
        Inject prompts that control overall structure and composition.
        
        Args:
            prompt: Prompt for structure (e.g., "castle silhouette, majestic architecture")
            weight: Injection weight
            resolution_levels: Which resolution levels to target
            sigma_start: Start of injection window
            sigma_end: End of injection window
            spatial_mask: Optional spatial mask
            
        Example:
            injector.add_structure_injection(
                "gothic cathedral silhouette, imposing architecture",
                weight=1.5,
                resolution_levels=["lowest"]
            )
        """
        logger.debug(f"Adding structure injection: '{prompt}' at {resolution_levels} resolution")
        
        # Use cached mappings for better performance
        target_blocks = []
        for resolution in resolution_levels:
            blocks = self._resolution_blocks_cache.get(resolution, [])
            target_blocks.extend(blocks)
        
        if not target_blocks:
            logger.warning(f"No blocks found for resolution levels: {resolution_levels}")
            return
        
        # Batch operation
        self._add_prompt_to_blocks(prompt, target_blocks, weight, sigma_start, sigma_end, spatial_mask)
        logger.debug(f"Applied structure injection to {len(target_blocks)} blocks")

    def add_midlevel_injection(self,
                             prompt: str,
                             weight: float = 1.0,
                             resolution_levels: List[str] = ["medium", "high"],
                             sigma_start: float = 0.0,
                             sigma_end: float = 1.0,
                             spatial_mask: Optional[torch.Tensor] = None):
        """
        Inject prompts that control medium-scale features.
        
        Args:
            prompt: Prompt for mid-level features (e.g., "ornate windows, decorative elements")
            weight: Injection weight
            resolution_levels: Which resolution levels to target
            sigma_start: Start of injection window
            sigma_end: End of injection window
            spatial_mask: Optional spatial mask
            
        Example:
            injector.add_midlevel_injection(
                "ornate gothic windows, flying buttresses, decorative spires",
                weight=1.3,
                resolution_levels=["medium"]
            )
        """
        logger.debug(f"Adding mid-level injection: '{prompt}' at {resolution_levels} resolution")
        
        # Use cached mappings for better performance
        target_blocks = []
        for resolution in resolution_levels:
            blocks = self._resolution_blocks_cache.get(resolution, [])
            target_blocks.extend(blocks)
        
        if not target_blocks:
            logger.warning(f"No blocks found for resolution levels: {resolution_levels}")
            return
        
        # Batch operation
        self._add_prompt_to_blocks(prompt, target_blocks, weight, sigma_start, sigma_end, spatial_mask)
        logger.debug(f"Applied mid-level injection to {len(target_blocks)} blocks")

    def add_hierarchical_prompts(self,
                               structure_prompt: str,
                               detail_prompt: str,
                               midlevel_prompt: Optional[str] = None,
                               weights: Optional[Dict[str, float]] = None,
                               sigma_start: float = 0.0,
                               sigma_end: float = 1.0):
        """
        Add a complete hierarchical prompt system with different conditioning at each scale.
        
        Args:
            structure_prompt: Low-resolution prompt for overall structure
            detail_prompt: High-resolution prompt for fine details
            midlevel_prompt: Optional mid-resolution prompt for medium features
            weights: Optional weight mapping for each level
            sigma_start: Start of injection window
            sigma_end: End of injection window
            
        Example:
            injector.add_hierarchical_prompts(
                structure_prompt="majestic gothic cathedral silhouette",
                midlevel_prompt="ornate stone arches, flying buttresses",
                detail_prompt="intricate stone carvings, weathered textures",
                weights={"structure": 1.5, "midlevel": 1.2, "detail": 1.0}
            )
        """
        logger.debug("Adding hierarchical multi-scale prompt system")
        
        # Balanced default weights for noticeable but controlled multi-scale effects
        default_weights = {"structure": 3.0, "midlevel": 2.5, "detail": 2.2}
        weights = {**default_weights, **(weights or {})}
        
        # OPTIMIZED: Direct batch operations instead of nested method calls
        
        # Structure (low resolution blocks)
        structure_blocks = []
        for resolution in ["low", "lowest"]:
            structure_blocks.extend(self._resolution_blocks_cache.get(resolution, []))
        
        if structure_blocks:
            self._add_prompt_to_blocks(
                structure_prompt, structure_blocks,
                weight=weights.get("structure", 1.0),
                sigma_start=sigma_start, sigma_end=sigma_end
            )
        
        # Mid-level (medium resolution blocks)
        if midlevel_prompt:
            midlevel_blocks = []
            for resolution in ["medium", "high"]:
                midlevel_blocks.extend(self._resolution_blocks_cache.get(resolution, []))
                
            if midlevel_blocks:
                self._add_prompt_to_blocks(
                    midlevel_prompt, midlevel_blocks,
                    weight=weights.get("midlevel", 1.0),
                    sigma_start=sigma_start, sigma_end=sigma_end
                )
        
        # Details (high resolution blocks)
        detail_blocks = []
        for resolution in ["highest", "high"]:
            detail_blocks.extend(self._resolution_blocks_cache.get(resolution, []))
            
        if detail_blocks:
            self._add_prompt_to_blocks(
                detail_prompt, detail_blocks,
                weight=weights.get("detail", 1.0),
                sigma_start=sigma_start, sigma_end=sigma_end
            )
        
        total_blocks = len(structure_blocks) + len(detail_blocks) + (len(midlevel_blocks) if midlevel_prompt else 0)
        logger.debug(f"Hierarchical prompt system configured: {total_blocks} total blocks")

    def add_stage_based_injection(self,
                                downsample_prompt: Optional[str] = None,
                                bottleneck_prompt: Optional[str] = None,
                                upsample_prompt: Optional[str] = None,
                                weights: Optional[Dict[str, float]] = None,
                                sigma_start: float = 0.0,
                                sigma_end: float = 1.0):
        """
        Add prompts based on processing stages rather than resolution levels.
        
        Args:
            downsample_prompt: Prompt for downsampling stage (structure formation)
            bottleneck_prompt: Prompt for bottleneck stage (global context)
            upsample_prompt: Prompt for upsampling stage (detail refinement)
            weights: Optional weight mapping for each stage
            sigma_start: Start of injection window
            sigma_end: End of injection window
            
        Example:
            injector.add_stage_based_injection(
                downsample_prompt="establish architectural structure",
                bottleneck_prompt="gothic cathedral global composition", 
                upsample_prompt="refine stone textures and details",
                weights={"downsample": 1.2, "bottleneck": 1.5, "upsample": 1.0}
            )
        """
        logger.debug("Adding stage-based injection system")
        
        weights = weights or {}
        total_blocks = 0
        
        # Use cached mappings for better performance
        if downsample_prompt:
            blocks = self._stage_blocks_cache.get("downsample", [])
            if blocks:
                self._add_prompt_to_blocks(
                    downsample_prompt, blocks,
                    weight=weights.get("downsample", 1.0),
                    sigma_start=sigma_start, sigma_end=sigma_end
                )
                total_blocks += len(blocks)
        
        if bottleneck_prompt:
            blocks = self._stage_blocks_cache.get("bottleneck", [])
            if blocks:
                self._add_prompt_to_blocks(
                    bottleneck_prompt, blocks,
                    weight=weights.get("bottleneck", 1.0),
                    sigma_start=sigma_start, sigma_end=sigma_end
                )
                total_blocks += len(blocks)
        
        if upsample_prompt:
            blocks = self._stage_blocks_cache.get("upsample", [])
            if blocks:
                self._add_prompt_to_blocks(
                    upsample_prompt, blocks,
                    weight=weights.get("upsample", 1.0),
                    sigma_start=sigma_start, sigma_end=sigma_end
                )
                total_blocks += len(blocks)
        
        logger.debug(f"Stage-based injection system configured: {total_blocks} total blocks")

    def add_extreme_multi_scale(self,
                               base_prompt: str,
                               structure_keywords: str = "massive, towering, imposing",
                               detail_keywords: str = "intricate, ornate, weathered",
                               weight_multiplier: float = 2.0):
        """
        Add EXTREME multi-scale injection for dramatic effects.
        
        This method creates very strong multi-scale prompts with high weights
        designed to dramatically override the base prompt.
        
        Args:
            base_prompt: Base subject (e.g., "cathedral", "castle", "mountain")
            structure_keywords: Structural descriptors for composition
            detail_keywords: Detail descriptors for textures/features
            weight_multiplier: Multiplier for already-strong default weights
            
        Example:
            injector.add_extreme_multi_scale(
                base_prompt="cathedral",
                structure_keywords="colossal gothic fortress",
                detail_keywords="carved gargoyles, ancient stonework"
            )
        """
        logger.info("Adding EXTREME multi-scale injection for dramatic effects")
        
        # Create extreme prompts
        structure_prompt = f"{structure_keywords} {base_prompt}, dramatic imposing architecture"
        detail_prompt = f"{detail_keywords} {base_prompt} details, elaborate craftsmanship"
        
        # Apply with strong but controlled weights
        extreme_weights = {
            "structure": 3.5 * weight_multiplier,
            "detail": 2.8 * weight_multiplier
        }
        
        self.add_hierarchical_prompts(
            structure_prompt=structure_prompt,
            detail_prompt=detail_prompt,
            weights=extreme_weights,
            sigma_start=1.0,  # Full diffusion range for maximum influence
            sigma_end=0.1
        )
        
        logger.info(f"EXTREME multi-scale configured: structure weight {extreme_weights['structure']}, detail weight {extreme_weights['detail']}")

    def get_resolution_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of which blocks correspond to which resolution levels.
        
        Returns:
            Dictionary mapping resolution levels to block identifier strings (for compatibility)
        """
        # Convert BlockIdentifier objects back to strings for API compatibility
        return {
            resolution: [str(block) for block in blocks] 
            for resolution, blocks in self._resolution_blocks_cache.items()
        }

    def get_stage_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of which blocks correspond to which processing stages.
        
        Returns:
            Dictionary mapping stages to block identifier strings (for compatibility)
        """
        # Convert BlockIdentifier objects back to strings for API compatibility
        return {
            stage: [str(block) for block in blocks] 
            for stage, blocks in self._stage_blocks_cache.items()
        }
