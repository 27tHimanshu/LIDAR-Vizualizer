import os
import numpy as np
import torch
from lidar_loader import load_lidar_file, preprocess_pointcloud, create_pillars
from model_inference import PointPillarsNet
from visualizer import LidarVisualizer
from traffic_analyzer import TrafficAnalyzer

def main():
    try:
        # Initialize model and visualizer
        model = PointPillarsNet()
        visualizer = LidarVisualizer()
        analyzer = TrafficAnalyzer()
        
        # Load LiDAR file
        lidar_file = "data/kitti/training/velodyne/000002.bin"
        if not os.path.exists(lidar_file):
            print(f"Error: LiDAR file {lidar_file} not found!")
            return
            
        # Load model weights
        weights_file = "pointPillar_model/pointpillar_7728.pth"
        if not os.path.exists(weights_file):
            print(f"Error: Model weights file {weights_file} not found!")
            return
            
        # Process point cloud
        print("Loading point cloud...")
        points = load_lidar_file(lidar_file)
        print(f"Loaded {len(points)} points")
        
        # Preprocess points
        print("Preprocessing points...")
        processed_points = preprocess_pointcloud(points)
        print(f"After preprocessing: {len(processed_points)} points")
        
        # Create pillars
        print("Creating pillars...")
        pillars, indices = create_pillars(processed_points)
        print(f"Created pillars shape: {pillars.shape}, indices shape: {indices.shape}")
        
        # Run inference
        print("Running inference...")
        boxes, scores, labels = model.detect(pillars, indices)
        print(f"Detected {len(boxes)} objects")
        
        # Filter predictions
        score_threshold = 0.3
        boxes, scores, labels = model.filter_predictions(boxes, scores, labels, score_threshold)
        print(f"After filtering (threshold={score_threshold}): {len(boxes)} objects")
        
        # Analyze traffic
        print("\nAnalyzing traffic patterns...")
        analysis = analyzer.analyze_traffic(boxes, scores, labels)
        report = analyzer.generate_report(analysis)
        print(report)
        
        # Visualize results
        print("\nVisualizing results...")
        visualizer.visualize(processed_points, boxes, labels, scores, save=True)
        input("\nPress Enter to close visualization...")
        visualizer.close()
        
        print("\nDone!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'visualizer' in locals():
            visualizer.close()

if __name__ == "__main__":
    main() 