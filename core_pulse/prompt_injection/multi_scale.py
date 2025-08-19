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
        
        # Log available resolution levels
        for block_id, info in self.patcher.block_mapper.resolution_levels.items():
            logger.debug(f"Block {block_id}: {info['resolution']} ({info['stage']})")

    def add_detail_injection(self,
                           prompt: str,
                           weight: float = 1.0,
                           resolution_levels: List[str] = ["highest", "high"],
                           sigma_start: float = 0.0,
                           sigma_end: float = 1.0,
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
        
        logger.debug(f"Applied detail injection to {len(target_blocks)} blocks: {target_blocks}")

    def add_structure_injection(self,
                              prompt: str,
                              weight: float = 1.0,
                              resolution_levels: List[str] = ["low", "lowest"],
                              sigma_start: float = 0.0,
                              sigma_end: float = 1.0,
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
        
        logger.debug(f"Applied structure injection to {len(target_blocks)} blocks: {target_blocks}")

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
        logger.info(f"Adding mid-level injection: '{prompt}' at {resolution_levels} resolution")
        
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
        
        logger.debug(f"Applied mid-level injection to {len(target_blocks)} blocks: {target_blocks}")

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
        logger.info("Adding hierarchical multi-scale prompt system")
        
        weights = weights or {}
        
        # Structure (low resolution)
        self.add_structure_injection(
            structure_prompt,
            weight=weights.get("structure", 1.0),
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )
        
        # Mid-level (medium resolution)
        if midlevel_prompt:
            self.add_midlevel_injection(
                midlevel_prompt,
                weight=weights.get("midlevel", 1.0),
                sigma_start=sigma_start,
                sigma_end=sigma_end
            )
        
        # Details (high resolution)
        self.add_detail_injection(
            detail_prompt,
            weight=weights.get("detail", 1.0),
            sigma_start=sigma_start,
            sigma_end=sigma_end
        )
        
        logger.info("Hierarchical prompt system configured")

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
        logger.info("Adding stage-based injection system")
        
        weights = weights or {}
        
        if downsample_prompt:
            blocks = self.patcher.block_mapper.get_blocks_by_stage("downsample")
            for block in blocks:
                self.add_injection(
                    block=block,
                    prompt=downsample_prompt,
                    weight=weights.get("downsample", 1.0),
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
        
        if bottleneck_prompt:
            blocks = self.patcher.block_mapper.get_blocks_by_stage("bottleneck")
            for block in blocks:
                self.add_injection(
                    block=block,
                    prompt=bottleneck_prompt,
                    weight=weights.get("bottleneck", 1.0),
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
        
        if upsample_prompt:
            blocks = self.patcher.block_mapper.get_blocks_by_stage("upsample")
            for block in blocks:
                self.add_injection(
                    block=block,
                    prompt=upsample_prompt,
                    weight=weights.get("upsample", 1.0),
                    sigma_start=sigma_start,
                    sigma_end=sigma_end
                )
        
        logger.info("Stage-based injection system configured")

    def get_resolution_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of which blocks correspond to which resolution levels.
        
        Returns:
            Dictionary mapping resolution levels to block identifiers
        """
        summary = {}
        for resolution in ["highest", "high", "medium", "low", "lowest"]:
            blocks = self.patcher.block_mapper.get_blocks_by_resolution(resolution)
            if blocks:
                summary[resolution] = blocks
        return summary

    def get_stage_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of which blocks correspond to which processing stages.
        
        Returns:
            Dictionary mapping stages to block identifiers
        """
        summary = {}
        for stage in ["downsample", "bottleneck", "upsample"]:
            blocks = self.patcher.block_mapper.get_blocks_by_stage(stage)
            if blocks:
                summary[stage] = blocks
        return summary


# Convenience functions for common multi-scale patterns
def create_hierarchical_architecture(pipeline: DiffusionPipeline,
                                   building_type: str = "castle",
                                   detail_level: str = "high") -> MultiScaleInjector:
    """
    Create a hierarchical architecture with structure, features, and details.
    
    Args:
        pipeline: Diffusion pipeline
        building_type: Type of architecture (castle, cathedral, etc.)
        detail_level: Level of detail (low, medium, high)
    """
    injector = MultiScaleInjector(pipeline)
    
    structure_prompts = {
        "castle": "majestic medieval castle silhouette, imposing fortress",
        "cathedral": "gothic cathedral silhouette, soaring spires",
        "palace": "grand palace silhouette, symmetrical architecture"
    }
    
    detail_prompts = {
        "low": f"simple {building_type} features",
        "medium": f"ornate {building_type} details, decorative elements", 
        "high": f"intricate {building_type} stonework, elaborate carvings, weathered textures"
    }
    
    injector.add_hierarchical_prompts(
        structure_prompt=structure_prompts.get(building_type, f"{building_type} silhouette"),
        midlevel_prompt=f"ornate {building_type} architectural features",
        detail_prompt=detail_prompts[detail_level]
    )
    
    return injector


def enhance_texture_detail(pipeline: DiffusionPipeline,
                         texture_prompt: str = "intricate textures, fine details",
                         detail_weight: float = 1.3) -> MultiScaleInjector:
    """
    Enhance texture detail at high resolution levels.
    """
    injector = MultiScaleInjector(pipeline)
    injector.add_detail_injection(texture_prompt, weight=detail_weight)
    return injector


def improve_composition_structure(pipeline: DiffusionPipeline,
                                composition_prompt: str = "balanced composition, harmonious structure",
                                structure_weight: float = 1.4) -> MultiScaleInjector:
    """
    Improve overall composition and structure at low resolution levels.
    """
    injector = MultiScaleInjector(pipeline)
    injector.add_structure_injection(composition_prompt, weight=structure_weight)
    return injector
