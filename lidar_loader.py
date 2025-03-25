import numpy as np
import os

def load_lidar_file(file_path):
    """
    Load KITTI LiDAR point cloud from binary file.
    Args:
        file_path: Path to the .bin file
    Returns:
        points: Nx4 array where N is number of points and 4 represents (x,y,z,intensity)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LiDAR file not found at: {file_path}")
        
    # Load point cloud from .bin file
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def preprocess_pointcloud(points, x_range=(0, 70.4), y_range=(-40, 40), z_range=(-3, 1)):
    """
    Preprocess the point cloud for PointPillars model.
    Args:
        points: Nx4 array of point cloud data
        x_range: Range of x coordinates to keep
        y_range: Range of y coordinates to keep
        z_range: Range of z coordinates to keep
    Returns:
        processed_points: Processed and filtered point cloud
    """
    # Remove points outside detection range
    mask = np.where((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
                    (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
                    (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1]))[0]
    
    points = points[mask]
    
    # Normalize intensity values
    points[:, 3] = np.clip(points[:, 3], 0, 1)
    
    return points

def create_pillars(points, max_points_per_pillar=100, max_pillars=12000,
                  x_step=0.16, y_step=0.16, z_step=4,
                  x_range=(0, 70.4), y_range=(-40, 40), z_range=(-3, 1)):
    """
    Convert point cloud to pillar representation.
    Args:
        points: Nx4 array of point cloud data
        max_points_per_pillar: Maximum number of points in each pillar
        max_pillars: Maximum number of pillars
        x_step, y_step, z_step: Size of pillars in each dimension
        x_range, y_range, z_range: Range of points to consider
    Returns:
        pillars: Array of pillar features
        indices: Array of pillar indices
    """
    if len(points) == 0:
        # Handle empty point cloud
        empty_pillars = np.zeros((1, max_points_per_pillar, 8))
        empty_indices = np.zeros((1, 2), dtype=np.int32)
        return empty_pillars, empty_indices

    # Calculate grid size
    x_size = int((x_range[1] - x_range[0]) / x_step)
    y_size = int((y_range[1] - y_range[0]) / y_step)
    
    # Initialize pillars and indices lists
    pillars = []
    indices = []
    
    # Calculate pillar indices for all points at once
    x_indices = ((points[:, 0] - x_range[0]) / x_step).astype(np.int32)
    y_indices = ((points[:, 1] - y_range[0]) / y_step).astype(np.int32)
    
    # Create unique pillar IDs
    pillar_ids = x_indices * y_size + y_indices
    unique_pillar_ids = np.unique(pillar_ids)
    
    # Process each unique pillar
    for pillar_id in unique_pillar_ids:
        if len(pillars) >= max_pillars:
            break
            
        # Get points in current pillar
        mask = pillar_ids == pillar_id
        pillar_points = points[mask]
        
        if len(pillar_points) > 0:
            # Calculate pillar center
            center = pillar_points.mean(axis=0)
            
            # Select points if too many
            if len(pillar_points) > max_points_per_pillar:
                distances = np.linalg.norm(pillar_points[:, :3] - center[:3], axis=1)
                indices_in_pillar = np.argsort(distances)[:max_points_per_pillar]
                pillar_points = pillar_points[indices_in_pillar]
            
            # Calculate features
            center_offset = pillar_points[:, :3] - center[:3]
            distances = np.linalg.norm(center_offset, axis=1, keepdims=True)
            
            # Combine features
            features = np.concatenate([
                pillar_points,  # original points (x,y,z,intensity)
                center_offset,  # offset from center (dx,dy,dz)
                distances      # distance to center
            ], axis=1)
            
            # Pad if necessary
            if len(features) < max_points_per_pillar:
                pad_size = max_points_per_pillar - len(features)
                padding = np.zeros((pad_size, features.shape[1]))
                features = np.vstack([features, padding])
            
            # Add to lists
            pillars.append(features)
            x_idx = pillar_id // y_size
            y_idx = pillar_id % y_size
            indices.append([x_idx, y_idx])
    
    # Handle case when no pillars were created
    if len(pillars) == 0:
        empty_pillars = np.zeros((1, max_points_per_pillar, 8))
        empty_indices = np.zeros((1, 2), dtype=np.int32)
        return empty_pillars, empty_indices
    
    # Convert lists to arrays
    pillars = np.stack(pillars)
    indices = np.stack(indices)
    
    # Pad to max_pillars if necessary
    if len(pillars) < max_pillars:
        pad_pillars = np.zeros((max_pillars - len(pillars), max_points_per_pillar, 8))
        pad_indices = np.zeros((max_pillars - len(indices), 2), dtype=np.int32)
        pillars = np.concatenate([pillars, pad_pillars], axis=0)
        indices = np.concatenate([indices, pad_indices], axis=0)
    else:
        pillars = pillars[:max_pillars]
        indices = indices[:max_pillars]
    
    return pillars, indices 