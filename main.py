import os
import argparse
from lidar_loader import load_lidar_file, preprocess_pointcloud, create_pillars
from model_inference import PointPillarsNet
from visualizer import LidarVisualizer

def main(args):
    # Load and preprocess point cloud
    print("Loading point cloud...")
    points = load_lidar_file(args.lidar_file)
    processed_points = preprocess_pointcloud(points)
    
    # Create pillars
    print("Creating pillars...")
    pillars, pillar_indices = create_pillars(processed_points)
    
    # Initialize model
    print("Loading model...")
    model = PointPillarsNet(args.model_path)
    
    # Run inference
    print("Running inference...")
    boxes, scores, labels = model.detect(pillars, pillar_indices)
    
    # Filter predictions
    boxes, scores, labels = model.filter_predictions(boxes, scores, labels, 
                                                   score_threshold=args.score_threshold)
    
    # Visualize results
    print("Visualizing results...")
    visualizer = LidarVisualizer()
    visualizer.visualize(processed_points, boxes, scores, labels)
    
    print(f"Detected {len(boxes)} objects:")
    class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"Object {i+1}: {class_names[label]} (confidence: {score:.2f})")
    
    input("Press Enter to close visualization...")
    visualizer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Object Detection from LiDAR")
    parser.add_argument("--lidar_file", type=str, required=True,
                        help="Path to KITTI format .bin file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained PointPillars model")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Confidence score threshold for filtering detections")
    
    args = parser.parse_args()
    main(args) 