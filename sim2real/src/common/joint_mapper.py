import numpy as np
from typing import List, Tuple, Optional, Dict


class JointMapper:
    """
    Maps between different joint-name spaces.
    Handles action/state conversion and parameter mapping.
    """
    
    def __init__(self, from_joint_names: List[str], to_joint_names: List[str]):
        """
        Initialize mapper between two joint spaces.
        
        Args:
            from_joint_names: Source joint names
            to_joint_names: Target joint names
        """
        self.from_names = from_joint_names
        self.to_names = to_joint_names
        self.from_to_idx, self.to_from_idx = self._compute_mapping()
        
    def _compute_mapping(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute index mappings between joint spaces.
        
        Returns:
            from_to_idx: Indices to map from source to target (-1 if joint not present)
            to_from_idx: Indices to map from target to source (-1 if joint not present)
        """
        # Create name to index mappings
        from_name_to_idx = {name: i for i, name in enumerate(self.from_names)}
        to_name_to_idx = {name: i for i, name in enumerate(self.to_names)}
        
        # Map from source to target
        from_to_idx = np.full(len(self.from_names), -1, dtype=int)
        for i, from_name in enumerate(self.from_names):
            if from_name in to_name_to_idx:
                from_to_idx[i] = to_name_to_idx[from_name]
        
        # Map from target to source
        to_from_idx = np.full(len(self.to_names), -1, dtype=int)
        for i, to_name in enumerate(self.to_names):
            if to_name in from_name_to_idx:
                to_from_idx[i] = from_name_to_idx[to_name]
                
        return from_to_idx, to_from_idx
    
    def map_action_from_to(self, action_from: np.ndarray, 
                          default_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Map action from source space to target space.
        
        Args:
            action_from: Action in source space
            default_values: Default values for joints not present in source (optional)
            
        Returns:
            action_to: Action in target space
        """
        action_from = np.asarray(action_from)
        
        if default_values is None:
            default_values = np.zeros(len(self.to_names))
        else:
            default_values = np.asarray(default_values)
        
        action_to = default_values.copy()
        
        # Copy values where mapping exists
        valid_mapping = self.from_to_idx >= 0
        from_indices = np.arange(len(self.from_names))[valid_mapping]
        to_indices = self.from_to_idx[valid_mapping]
        
        action_to[to_indices] = action_from[from_indices]
        
        return action_to
    
    def map_state_to_from(self, state_to: np.ndarray) -> np.ndarray:
        """
        Map state from target space back to source space.
        
        Args:
            state_to: State in target space
            
        Returns:
            state_from: State in source space
        """
        state_to = np.asarray(state_to)
        state_from = np.zeros(len(self.from_names))
        
        # Copy values where mapping exists
        valid_mapping = self.to_from_idx >= 0
        to_indices = np.arange(len(self.to_names))[valid_mapping]
        from_indices = self.to_from_idx[valid_mapping]
        
        state_from[from_indices] = state_to[to_indices] 
        
        return state_from
    
    def map_parameters_to_from(self, params_to: np.ndarray) -> np.ndarray:
        """
        Map parameters (kp, kd, limits, etc.) from target space to source space.
        
        Args:
            params_to: Parameters in target space
            
        Returns:
            params_from: Parameters in source space (zero for unmapped joints)
        """
        return self.map_state_to_from(params_to)
    
    def get_valid_mapping_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get boolean masks indicating which joints have valid mappings.
        
        Returns:
            from_mask: Boolean mask for source joints that have mapping to target
            to_mask: Boolean mask for target joints that have mapping from source
        """
        from_mask = self.from_to_idx >= 0
        to_mask = self.to_from_idx >= 0
        return from_mask, to_mask
    
    def get_mapping_info(self) -> Dict:
        """
        Get detailed mapping information for debugging.
        
        Returns:
            Dict with mapping details
        """
        from_mask, to_mask = self.get_valid_mapping_mask()
        
        mapped_from = [self.from_names[i] for i in range(len(self.from_names)) if from_mask[i]]
        mapped_to = [self.to_names[i] for i in range(len(self.to_names)) if to_mask[i]]
        
        unmapped_from = [self.from_names[i] for i in range(len(self.from_names)) if not from_mask[i]]
        unmapped_to = [self.to_names[i] for i in range(len(self.to_names)) if not to_mask[i]]
        
        return {
            'from_space_size': len(self.from_names),
            'to_space_size': len(self.to_names),
            'mapped_joints': len(mapped_from),
            'mapped_from_joints': mapped_from,
            'mapped_to_joints': mapped_to,
            'unmapped_from_joints': unmapped_from,
            'unmapped_to_joints': unmapped_to,
        }

def create_name_mapper(from_joint_names: List[str], to_joint_names: List[str]) -> JointMapper:
    return JointMapper(from_joint_names, to_joint_names)
