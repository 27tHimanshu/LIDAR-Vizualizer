import torch
import torch.nn as nn
import numpy as np

class MinimalPointPillars(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Minimal feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),  # Added another layer for better feature extraction
            nn.ReLU()
        )
        
        # Minimal backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Detection heads with better initialization
        self.conv_cls = nn.Conv2d(128, num_classes, 1)
        self.conv_box = nn.Conv2d(128, 7, 1)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to prevent clustering of confidence scores"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, pillars, indices):
        batch_size = pillars.size(0)
        num_pillars = pillars.size(1)
        num_points = pillars.size(2)
        
        # Reshape pillars for feature extraction
        pillars_flat = pillars.reshape(-1, pillars.size(-1))  # (B*N*P, 8)
        
        # Extract features
        point_features = self.feature_net(pillars_flat)  # (B*N*P, 32)
        point_features = point_features.view(batch_size, num_pillars, num_points, -1)  # (B, N, P, 32)
        
        # Max pool over points in each pillar
        pillar_features = point_features.max(dim=2)[0]  # (B, N, 32)
        
        # Create pseudo-image
        canvas_size = (batch_size, 32, 50, 50)
        canvas = torch.zeros(canvas_size, dtype=torch.float32, device=pillars.device)
        
        # Scatter features to canvas
        for b in range(batch_size):
            x = torch.clamp(indices[b, :, 0].long(), 0, canvas_size[2]-1)
            y = torch.clamp(indices[b, :, 1].long(), 0, canvas_size[3]-1)
            features = pillar_features[b].permute(1, 0)  # (32, N)
            canvas[b, :, x, y] = features[:, :num_pillars]
        
        # Process through backbone
        x = self.backbone(canvas)
        
        # Generate predictions with temperature scaling for better confidence distribution
        cls_logits = self.conv_cls(x)
        cls_preds = cls_logits / 2.0  # Temperature scaling to spread out confidence scores
        box_preds = self.conv_box(x)
        
        return cls_preds, box_preds

class PointPillarsNet:
    def __init__(self, model_path=None):
        self.device = torch.device('cpu')
        self.model = MinimalPointPillars().to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def detect(self, pillars, pillar_indices):
        try:
            # Convert inputs
            pillars = torch.from_numpy(pillars).float().to(self.device)
            indices = torch.from_numpy(pillar_indices).long().to(self.device)
            
            # Ensure proper dimensions
            if len(pillars.shape) == 2:
                pillars = pillars.unsqueeze(0).unsqueeze(2)  # Add batch and point dimensions
            elif len(pillars.shape) == 3:
                pillars = pillars.unsqueeze(0)  # Add batch dimension
            
            # Handle indices properly based on input shape
            if len(indices.shape) == 2:  # (N, 2) shape
                indices = indices.unsqueeze(0)  # Add batch dimension: (1, N, 2)
            
            # Run inference
            cls_preds, box_preds = self.model(pillars, indices)
            
            # Post-process with better confidence calculation
            cls_scores = torch.sigmoid(cls_preds)  # (B, num_classes, H, W)
            
            # Reshape predictions
            B, C, H, W = cls_scores.shape
            cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, num_classes)
            box_preds = box_preds.permute(0, 2, 3, 1).reshape(-1, 7)  # (B*H*W, 7)
            
            # Get best class scores and labels with confidence adjustment
            scores, labels = torch.max(cls_scores, dim=1)  # (B*H*W,)
            
            # Apply additional score normalization
            score_threshold = 0.3
            normalized_scores = (scores - score_threshold) / (1 - score_threshold)
            normalized_scores = torch.clamp(normalized_scores, 0, 1)
            
            # Convert to numpy
            boxes = box_preds.cpu().numpy()
            scores = normalized_scores.cpu().numpy()
            labels = labels.cpu().numpy()
            
            return boxes, scores, labels
            
        except Exception as e:
            print(f"Error in detect(): {str(e)}")
            raise Exception(f"Inference failed: {str(e)}")
    
    def filter_predictions(self, boxes, scores, labels, score_threshold=0.3):
        # Apply non-maximum suppression to remove duplicate detections
        keep_indices = []
        for class_id in range(3):  # For each class
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            if len(class_boxes) > 0:
                # Sort by confidence
                sorted_indices = np.argsort(-class_scores)
                sorted_boxes = class_boxes[sorted_indices]
                sorted_scores = class_scores[sorted_indices]
                
                # Keep track of which boxes to keep
                keep = np.ones(len(sorted_boxes), dtype=bool)
                
                # Calculate centers
                centers = sorted_boxes[:, :2]  # Use x,y coordinates
                
                # Remove duplicates based on center distance
                for i in range(len(sorted_boxes)):
                    if keep[i]:
                        # Calculate distance to all other centers
                        distances = np.linalg.norm(centers[i+1:] - centers[i], axis=1)
                        # Mark boxes that are too close as duplicates
                        duplicates = distances < 0.5  # 0.5 meter threshold
                        keep[i+1:][duplicates] = False
                
                keep_indices.extend(np.where(class_mask)[0][sorted_indices[keep]])
        
        # Apply confidence threshold and keep non-duplicate detections
        keep_indices = np.array(keep_indices)
        mask = scores[keep_indices] > score_threshold
        
        filtered_boxes = boxes[keep_indices][mask]
        filtered_scores = scores[keep_indices][mask]
        filtered_labels = labels[keep_indices][mask]
        
        return filtered_boxes, filtered_scores, filtered_labels 