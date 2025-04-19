# import os
# import argparse
# from lidar_loader import load_lidar_file, preprocess_pointcloud, create_pillars
# from model_inference import PointPillarsNet
# from visualizer import LidarVisualizer
# from traffic_analyzer import TrafficAnalyzer

# def load_and_preprocess_lidar(lidar_file):
#     print(f"Loading point cloud from: {lidar_file}...")
#     points = load_lidar_file(lidar_file)
#     print(f"Loaded {len(points)} points.")
    
#     print("Preprocessing points...")
#     processed_points = preprocess_pointcloud(points)
#     print(f"After preprocessing: {len(processed_points)} points.")
    
#     return processed_points


# def run_inference_and_filter(model, pillars, pillar_indices, score_threshold):
#     print("Running inference...")
#     boxes, scores, labels = model.detect(pillars, pillar_indices)
    
#     print("Filtering predictions...")
#     boxes, scores, labels = model.filter_predictions(boxes, scores, labels, score_threshold)
#     return boxes, scores, labels


# def visualize_and_analyze(processed_points, boxes, scores, labels):
#     visualizer = LidarVisualizer()
    
#     print("\nVisualizing results...")
#     visualizer.visualize(processed_points, boxes, scores, labels)


    
#     print(f"Detected {len(boxes)} objects:")
#     class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    
#     # Print detailed object information
#     for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
#         print(f"Object {i + 1}: {class_names[label]} (confidence: {score:.2f})")
    
#     # Traffic Analysis
#     analyzer = TrafficAnalyzer()
#     print("\nAnalyzing traffic patterns...")
#     analysis = analyzer.analyze_traffic(boxes, scores, labels)
#     report = analyzer.generate_report(analysis)
#     print("\nTraffic Analysis Report:")
#     print(report)
    
#     input("Press Enter to close visualization...")
#     visualizer.close()


# def main():
#     print("Welcome to the LiDAR Object Detection and Traffic Analysis System")
#     print("--------------------------------------------------------------")
    
#     # Prompt for file selection
#     print("\nSelect a LiDAR file:")
#     lidar_files = ["000000", "000001", "000002", "000003", "000004", "000005"]
#     for i, file in enumerate(lidar_files):
#         print(f"{i+1}. {file}")
    
#     lidar_choice = int(input("\nEnter the number corresponding to the LiDAR file: "))
#     lidar_file = f"data/kitti/training/velodyne/{lidar_files[lidar_choice - 1]}.bin"
    
#     if not os.path.exists(lidar_file):
#         print(f"Error: LiDAR file {lidar_file} not found!")
#         return
    
#     # Prompt for score threshold
#     score_threshold = float(input("\nEnter the confidence score threshold (default 0.5): ") or 0.5)
    
#     print(f"\nRunning code with LiDAR file: {lidar_file} and score threshold: {score_threshold}...")
    
#     # Load and preprocess LiDAR data
#     processed_points = load_and_preprocess_lidar(lidar_file)
    
#     # Create pillars
#     print("Creating pillars...")
#     pillars, pillar_indices = create_pillars(processed_points)
    
#     # Initialize the model
#     print("Loading model...")
#     model = PointPillarsNet("pointPillar_model/pointpillar_7728.pth")
    
#     # Run inference and filter predictions
#     boxes, scores, labels = run_inference_and_filter(model, pillars, pillar_indices, score_threshold)
    
#     # Visualize results and analyze traffic
#     visualize_and_analyze(processed_points, boxes, scores, labels)


# if __name__ == "__main__":
#     main()

import os
import argparse
from lidar_loader import load_lidar_file, preprocess_pointcloud, create_pillars
from model_inference import PointPillarsNet
from visualizer import LidarVisualizer
from traffic_analyzer import TrafficAnalyzer

def load_and_preprocess_lidar(lidar_file):
    print(f"Loading point cloud from: {lidar_file}...")
    points = load_lidar_file(lidar_file)
    print(f"Loaded {len(points)} points.")
    
    print("Preprocessing points...")
    processed_points = preprocess_pointcloud(points)
    print(f"After preprocessing: {len(processed_points)} points.")
    
    return processed_points


def run_inference_and_filter(model, pillars, pillar_indices, score_threshold):
    print("Running inference...")
    boxes, scores, labels = model.detect(pillars, pillar_indices)
    
    print("Filtering predictions...")
    boxes, scores, labels = model.filter_predictions(boxes, scores, labels, score_threshold)
    return boxes, scores, labels


def visualize_and_analyze(processed_points, boxes, scores, labels):
    visualizer = LidarVisualizer()
    
    print("\nVisualizing results...")
    visualizer.visualize(processed_points, boxes, scores, labels)

    print(f"Detected {len(boxes)} objects:")
    class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    
    # Print detailed object information
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"Object {i + 1}: {class_names[label]} (confidence: {score:.2f})")
    
    # Traffic Analysis
    analyzer = TrafficAnalyzer()
    print("\nAnalyzing traffic patterns...")
    analysis = analyzer.analyze_traffic(boxes, scores, labels)
    report = analyzer.generate_report(analysis)
    print("\nTraffic Analysis Report:")
    print(report)
    
    input("Press Enter to close visualization...")
    visualizer.close()


def main():
    print("Welcome to the LiDAR Object Detection and Traffic Analysis System")
    print("--------------------------------------------------------------")
    
    # Prompt for file selection
    print("\nSelect a LiDAR file:")
    lidar_files = [f"{i:06d}" for i in range(20)]  # Generates 000000 to 000019
    for i, file in enumerate(lidar_files):
        print(f"{i+1:2d}. {file}")
    
    lidar_choice = int(input("\nEnter the number corresponding to the LiDAR file: "))
    if lidar_choice < 1 or lidar_choice > 20:
        print("Invalid selection! Please choose a number between 1 and 20.")
        return
    
    lidar_file = f"data/kitti/training/velodyne/{lidar_files[lidar_choice - 1]}.bin"
    
    if not os.path.exists(lidar_file):
        print(f"Error: LiDAR file {lidar_file} not found!")
        return
    
    # Prompt for score threshold
    score_threshold = float(input("\nEnter the confidence score threshold (default 0.5): ") or 0.5)
    
    print(f"\nRunning code with LiDAR file: {lidar_file} and score threshold: {score_threshold}...")
    
    # Load and preprocess LiDAR data
    processed_points = load_and_preprocess_lidar(lidar_file)
    
    # Create pillars
    print("Creating pillars...")
    pillars, pillar_indices = create_pillars(processed_points)
    
    # Initialize the model
    print("Loading model...")
    model = PointPillarsNet("pointPillar_model/pointpillar_7728.pth")
    
    # Run inference and filter predictions
    boxes, scores, labels = run_inference_and_filter(model, pillars, pillar_indices, score_threshold)
    
    # Visualize results and analyze traffic
    visualize_and_analyze(processed_points, boxes, scores, labels)


if __name__ == "__main__":
    main()