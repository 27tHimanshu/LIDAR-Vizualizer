import numpy as np
import open3d as o3d
import os

class LidarVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600, visible=True)  # Smaller window size
        
        # Set rendering options for better CPU performance
        opt = self.vis.get_render_option()
        opt.point_size = 2  # Slightly larger points for better visibility with fewer points
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_show_normal = False
        opt.light_on = False  # Disable lighting
        opt.mesh_show_wireframe = True  # Show wireframe for better performance
        opt.mesh_show_back_face = False  # Don't render back faces
        
        # Set view control for smoother interaction
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.3)
        view_control.set_lookat([0, 0, 0])
        view_control.set_front([0.5, -0.5, 0.5])
        view_control.set_up([0, 0, 1])

    def create_point_cloud(self, points):
        """
        Create Open3D point cloud object from numpy array with moderate downsampling.
        """
        # Less aggressive downsampling - show more points
        if len(points) > 30000:  # Increased from 10000 to 30000
            skip = len(points) // 30000
            points = points[::skip]
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Simplified coloring
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.clip(points[:, 3], 0, 1)  # Normalize intensity to [0,1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def create_bbox(self, box, color=(1, 0, 0)):
        """
        Create a bounding box line set from box parameters.
        box: [x, y, z, l, w, h, theta]
        """
        center = box[:3]
        dimensions = np.abs(box[3:6])  # Take absolute values of dimensions
        rotation = box[6]
        
        # Adjust Z-coordinate to move the box to the ground
        center[2] -= dimensions[2] / 2  # Move it down by half the height (box[5] is height)
        
        # Create box corners
        dx, dy, dz = dimensions[0]/2, dimensions[1]/2, dimensions[2]/2
        box_points = np.array([
            [dx, dy, dz], [dx, dy, -dz], [dx, -dy, dz], [dx, -dy, -dz],
            [-dx, dy, dz], [-dx, dy, -dz], [-dx, -dy, dz], [-dx, -dy, -dz],
        ])
        
        # Rotate box
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        box_points = np.dot(box_points, rotation_matrix.T)
        
        # Translate box
        box_points = box_points + center
        
        # Create line set with thicker lines for better visibility
        lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                 [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # Make lines more visible
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        
        return line_set

    def save_point_cloud(self, pcd, filename="point_cloud.ply"):
        """
        Save point cloud to PLY file
        """
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        save_path = os.path.join("output", filename)
        
        # Save the point cloud
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Point cloud saved to: {save_path}")
        return save_path

    def visualize(self, points, boxes=None, scores=None, labels=None, save=True, filename="point_cloud.ply"):
        """
        Visualize point cloud and detection results.
        """
        # Clear previous geometries
        self.vis.clear_geometries()
        
        # Create and add point cloud
        pcd = self.create_point_cloud(points)
        self.vis.add_geometry(pcd)
        
        # Save point cloud if requested
        if save:
            self.save_point_cloud(pcd, filename)
        
        # Add detection boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Default color: red for cars
                color = (1, 0, 0)
                
                # Check for specific object class
                if labels is not None:
                    label = int(labels[i]) if not isinstance(labels[i], int) else labels[i]
                    print(f"Casted Label[{i}]: {label}")

                    if label == 1:  # Pedestrian
                        print("Color: GREEN")
                        color = (0, 1, 0)
                    elif label == 2:  # Cyclist
                        print("Color: BLUE")
                        color = (0, 0, 1)
                    else:
                        print("Color: RED")
                        color = (1, 0, 0)
                # Create and add bounding box
                bbox = self.create_bbox(box, color)
                self.vis.add_geometry(bbox)
        
        print("\nVisualization window is open. Close the window to continue...")
        
        # Keep updating until window is closed
        while True:
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()

    def close(self):
        """
        Close the visualization window.
        """
        self.vis.destroy_window()
