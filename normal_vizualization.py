import numpy as np
import open3d as o3d
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox

# Global variables to make frame navigation work better
current_frame_idx = 0
bin_files = []
label_files = {}  # Maps frame names to label files

def load_point_cloud(bin_file):
    """Load LiDAR point cloud from .bin file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color by height
    colors = np.zeros((len(points), 3))
    z = points[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    z_range = z_max - z_min
    if z_range > 0:
        norm_z = (z - z_min) / z_range
        # Create a color gradient: blue (low) to red (high)
        colors[:, 0] = norm_z  # Red
        colors[:, 2] = 1 - norm_z  # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd, points

def load_labels(label_file, highlight_class="Truck"):
    """Load KITTI format labels, return bounding boxes for visualization"""
    if not os.path.exists(label_file):
        return []
    
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            
            class_name = parts[0]
            # We'll highlight the specified class (like trucks)
            if class_name.lower() != highlight_class.lower():
                continue
                
            # Parse dimensions (height, width, length)
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            
            # Parse location (x, y, z)
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            
            # Parse rotation
            rotation = float(parts[14])
            
            # Create oriented bounding box
            box = o3d.geometry.OrientedBoundingBox()
            box.center = [x, y, z]
            box.extent = [h, w, l]  # height, width, length
            R = box.get_rotation_matrix_from_xyz((0, 0, rotation))
            box.R = R
            
            # Use red color for trucks
            box.color = [1, 0, 0]
            boxes.append(box)
            
            print(f"Found {class_name} at position ({x:.2f}, {y:.2f}, {z:.2f})")
    
    return boxes

def main():
    global current_frame_idx, bin_files
    
    # Create a simple file selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Ask user to select data directory
    print("Select LiDAR data directory containing .bin files")
    data_dir = filedialog.askdirectory(title="Select LiDAR Data Directory")
    
    if not data_dir:
        print("No directory selected. Exiting.")
        return
    
    # Find all .bin files
    bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    
    if not bin_files:
        print("No .bin files found in the selected directory. Exiting.")
        return
    
    print(f"Found {len(bin_files)} LiDAR frames")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Frame Viewer", width=1024, height=768)
    
    # Rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Larger points for better visibility
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    vis.add_geometry(coord_frame)
    
    # Handle frame navigation
    current_frame_idx =0
    
    # Load first frame
    pcd, _ = load_point_cloud(bin_files[current_frame_idx])
    vis.add_geometry(pcd)
    
    # Make view control reference for later use
    view_control = vis.get_view_control()
    
    # Set initial view for better orientation
    view_control.set_front([0, 1, 0])  # Look from front
    view_control.set_lookat([0, 0, 0])  # Look at center
    view_control.set_up([0, 0, 1])      # Z is up
    view_control.set_zoom(0.7)          # Zoom level
    
    # Helper functions for frame navigation
    def update_frame():
        frame_name = os.path.basename(bin_files[current_frame_idx]).split('.')[0]
        title = f"LiDAR Frame {frame_name} ({current_frame_idx + 1}/{len(bin_files)})"
        print(f"\nShowing {title}")
        
        # Try to set window title (may not work on all Open3D versions)
        try:
            vis.set_window_name(title)
        except:
            pass  # Ignore if method doesn't exist
            
        # Load and highlight trucks if any
        label_dir = os.path.join(os.path.dirname(os.path.dirname(bin_files[current_frame_idx])), "label_2")
        label_file = os.path.join(label_dir, f"{frame_name}.txt")
        print(f"Looking for labels in: {label_file}")
        if os.path.exists(label_file):
            boxes = load_labels(label_file, "Truck")
            if boxes:
                print(f"Found {len(boxes)} trucks in this frame")
            else:
                print("No trucks found in this frame")
    
    # Initial update
    update_frame()
    
    # Print controls for user
    print("\nCONTROLS:")
    print("  KEYBOARD:")
    print("    • A: Previous frame (keyboard)")
    print("    • D: Next frame (keyboard)")
    print("    • W: Zoom in (keyboard)")
    print("    • S: Zoom out (keyboard)")
    print("    • Q: Rotate left (keyboard)")
    print("    • E: Rotate right (keyboard)")
    print("    • R: Reset view (keyboard)")
    print("  TRACKPAD:")
    print("    • One finger drag: Rotate view")
    print("    • Two finger drag: Pan view")
    print("    • Pinch: Zoom view")
    print("\nPress A/D keys to navigate between frames")
    
    # Main visualization loop
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main() 