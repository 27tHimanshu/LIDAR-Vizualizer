import numpy as np
import torch
from lidar_loader import load_lidar_file, preprocess_pointcloud, create_pillars
from model_inference import PointPillarsNet
from visualizer import LidarVisualizer
import os

def main():
    # Test with a specific file
    lidar_file = "data/kitti/testing/velodyne/004369.bin"  # Using a different file
    model_path = "pointPillar_model/pointpillar_7728.pth"
    
    # Check files exist
    if not os.path.exists(lidar_file):
        print(f"Error: LiDAR file not found at {lidar_file}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    try:
        # 1. Load point cloud
        print("Loading point cloud...")
        points = load_lidar_file(lidar_file)
        print(f"Loaded {len(points)} points")
        
        # 2. Preprocess points
        print("Preprocessing points...")
        processed_points = preprocess_pointcloud(points)
        print(f"After preprocessing: {len(processed_points)} points")
        
        # 3. Create pillars
        print("Creating pillars...")
        pillars, indices = create_pillars(processed_points)
        print(f"Created pillars shape: {pillars.shape}, indices shape: {indices.shape}")
        
        # 4. Initialize model
        print("Loading model...")
        model = PointPillarsNet(model_path)
        
        # 5. Run inference
        print("Running inference...")
        boxes, scores, labels = model.detect(pillars, indices)
        print(f"Detected {len(boxes)} objects")
        
        # 6. Filter predictions
        score_threshold = 0.3
        boxes, scores, labels = model.filter_predictions(boxes, scores, labels, score_threshold)
        print(f"After filtering (threshold={score_threshold}): {len(boxes)} objects")
        
        # 7. Visualize and save
        print("Visualizing results...")
        visualizer = LidarVisualizer()
        
        # Get base filename for saving
        base_filename = os.path.splitext(os.path.basename(lidar_file))[0]
        save_filename = f"{base_filename}_processed.ply"
        
        # Visualize and save point cloud
        visualizer.visualize(processed_points, boxes, scores, labels, save=True, filename=save_filename)
        print(f"\nPoint cloud will be saved as: output/{save_filename}")
        
        print("\nDetection Results:")
        class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            print(f"Object {i+1}: {class_names[label]} (confidence: {score:.2f})")
            print(f"Box: center=({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}), "
                  f"dims=({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}), rot={box[6]:.2f}")
        
        input("\nPress Enter to close visualization...")
        visualizer.close()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 