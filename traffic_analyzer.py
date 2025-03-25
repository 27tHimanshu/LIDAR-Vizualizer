import numpy as np
from collections import defaultdict

class TrafficAnalyzer:
    def __init__(self):
        self.class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        
    def analyze_traffic(self, boxes, scores, labels):
        """
        Analyze traffic patterns from detection results
        Args:
            boxes: Nx7 array of box parameters (x,y,z,l,w,h,theta)
            scores: N array of confidence scores
            labels: N array of class labels (0=Car, 1=Pedestrian, 2=Cyclist)
        """
        # Initialize analysis results
        analysis = {
            "total_objects": len(boxes),
            "class_counts": defaultdict(int),
            "class_percentages": {},
            "zone_analysis": defaultdict(int),
            "density_zones": {},
            "safety_metrics": {},
            "position_stats": defaultdict(list)
        }
        
        # Count objects by class
        for label in labels:
            analysis["class_counts"][self.class_names[label]] += 1
            
        # Calculate percentages
        total = len(labels)
        if total > 0:
            for class_name, count in analysis["class_counts"].items():
                analysis["class_percentages"][class_name] = (count / total) * 100
                
        # Analyze zones (left, center, right based on x-coordinate)
        for box, label in zip(boxes, labels):
            x = box[0]  # x-coordinate
            if x < -5:
                zone = "Left"
            elif x > 5:
                zone = "Right"
            else:
                zone = "Center"
            analysis["zone_analysis"][zone] += 1
            
            # Store positions for each class
            class_name = self.class_names[label]
            analysis["position_stats"][class_name].append(box[:3])  # store x,y,z
            
        # Calculate safety metrics
        if len(boxes) > 0:
            analysis["safety_metrics"] = self._calculate_safety_metrics(boxes, labels)
            
        # Determine density zones
        total_objects = sum(analysis["zone_analysis"].values())
        if total_objects > 0:
            for zone, count in analysis["zone_analysis"].items():
                percentage = (count / total_objects) * 100
                if percentage > 40:
                    density = "High traffic"
                elif percentage > 20:
                    density = "Medium traffic"
                else:
                    density = "Low traffic"
                analysis["density_zones"][zone] = density
        
        return analysis
    
    def _calculate_safety_metrics(self, boxes, labels):
        """Calculate safety-related metrics from detection results"""
        safety_metrics = {}
        
        # Calculate distances between objects of the same class
        for class_id in range(3):
            class_name = self.class_names[class_id]
            class_boxes = boxes[labels == class_id]
            
            if len(class_boxes) > 1:
                # Calculate distances between all pairs of this class
                distances = []
                for i in range(len(class_boxes)):
                    for j in range(i + 1, len(class_boxes)):
                        dist = np.linalg.norm(class_boxes[i, :3] - class_boxes[j, :3])
                        distances.append(dist)
                
                if distances:
                    safety_metrics[f"Min {class_name}-to-{class_name} distance"] = min(distances)
                    safety_metrics[f"Avg {class_name} spacing"] = sum(distances) / len(distances)
        
        # Calculate minimum distances between cars and pedestrians
        car_boxes = boxes[labels == 0]
        ped_boxes = boxes[labels == 1]
        
        if len(car_boxes) > 0 and len(ped_boxes) > 0:
            car_ped_distances = []
            for car_box in car_boxes:
                for ped_box in ped_boxes:
                    dist = np.linalg.norm(car_box[:3] - ped_box[:3])
                    car_ped_distances.append(dist)
            
            if car_ped_distances:
                safety_metrics["Min car-to-pedestrian distance"] = min(car_ped_distances)
        
        return safety_metrics
    
    def generate_report(self, analysis):
        """Generate a formatted report from analysis results"""
        report = []
        report.append("Traffic Analysis Report")
        report.append("---------------------")
        
        # Object counts
        report.append(f"Total Objects: {analysis['total_objects']}")
        for class_name, count in analysis['class_counts'].items():
            percentage = analysis['class_percentages'][class_name]
            report.append(f"- {class_name}s: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Density Zones
        report.append("Density Zones:")
        for zone, density in analysis['density_zones'].items():
            count = analysis['zone_analysis'][zone]
            report.append(f"- {zone} side: {density} ({count} objects)")
        report.append("")
        
        # Safety Metrics
        report.append("Safety Metrics:")
        for metric, value in analysis['safety_metrics'].items():
            report.append(f"- {metric}: {value:.1f}m")
        
        return "\n".join(report) 