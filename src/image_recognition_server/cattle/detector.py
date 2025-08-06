"""牛只检测器模块"""
import os
import sys
import numpy as np
import cv2
import requests
from typing import Dict, List, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import supervision as sv
    from ultralytics import YOLO
    DEPS_INSTALLED = True
except ImportError:
    logger.warning("缺少依赖项: supervision 或 ultralytics。牛只检测功能将不可用。")
    logger.warning("请安装依赖: pip install supervision ultralytics opencv-python")
    DEPS_INSTALLED = False


class CattleDetector:
    """牛只检测器类"""
    
    def __init__(self, model_path: str = None, confidence: float = 0.25, iou_threshold: float = 0.5):
        """
        初始化牛只检测器
        
        参数:
            model_path: YOLO模型路径，默认使用yolov8x.pt
            confidence: 检测置信度阈值，默认0.25
            iou_threshold: NMS的IoU阈值，默认0.5，值越大重叠检测越少
        """
        if not DEPS_INSTALLED:
            raise ImportError("缺少依赖项: supervision 或 ultralytics。请安装依赖后再使用牛只检测功能。")
        
        if model_path is None:
            # 尝试在常见位置查找模型
            model_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "yolo", "yolov8x.pt"),
                "yolov8x.pt"  # 如果本地没有，将从Ultralytics下载
            ]
            
            # 尝试加载可用的模型
            for path in model_paths:
                if path and (isinstance(path, str) and os.path.exists(path) or not isinstance(path, str)):
                    model_path = path
                    break
            else:
                model_path = "yolov8x.pt"  # 默认使用从Ultralytics下载的模型
        
        logger.info(f"加载YOLO模型: {model_path}")
        # 加载YOLO模型
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # 牛的类别ID (COCO数据集中cow的ID是19)
        self.cow_class_id = 19
    
    def detect(self, image_source: Union[str, np.ndarray], confidence: Optional[float] = None, iou_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        检测图像中的牛只
        
        参数:
            image_source: 图像路径、URL或numpy数组
            confidence: 检测置信度阈值，如不提供则使用初始化时设置的值
            iou_threshold: NMS的IoU阈值，如不提供则使用初始化时设置的值
            
        返回:
            包含检测结果的列表，每个检测结果包含:
            - box: [x1, y1, x2, y2] 边界框坐标
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
        """
        # 使用传入的参数或默认值
        conf = confidence if confidence is not None else self.confidence
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        # 加载图像
        if isinstance(image_source, str):
            image = self.load_image(image_source)
        else:
            image = image_source
            
        if image is None:
            raise ValueError(f"无法加载图像: {image_source}")
        
        # 执行检测，使用更高的IoU阈值进行NMS
        results = self.model(image, imgsz=1280, conf=conf, iou=iou, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 只保留牛类别的检测结果
        if len(detections) > 0:
            cow_mask = np.array([class_id == self.cow_class_id for class_id in detections.class_id])
            detections = detections[cow_mask]
            
            # 计算图像面积并过滤掉过大的检测框（可能是误检）
            if len(detections) > 0:
                height, width = image.shape[:2]
                image_area = height * width
                area_mask = (detections.area / image_area) < 0.7
                detections = detections[area_mask]
                
                # 应用自定义的重叠检测合并
                detections = self._merge_overlapping_detections(detections)
        
        # 转换为易于理解的字典列表
        detection_results = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            detection_results.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(detections.confidence[i]),
                "class_id": int(detections.class_id[i]),
                "class_name": results.names[detections.class_id[i]],
                "area": float(detections.area[i])
            })
            
        return detection_results
    
    def _merge_overlapping_detections(self, detections: sv.Detections, overlap_threshold: float = 0.7) -> sv.Detections:
        """
        合并重叠度高的检测框
        
        参数:
            detections: 检测结果
            overlap_threshold: 重叠度阈值，超过此值的检测框将被合并
            
        返回:
            处理后的检测结果
        """
        if len(detections) <= 1:
            return detections
        
        # 计算检测框之间的IoU矩阵
        boxes = detections.xyxy
        n = len(boxes)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                iou_matrix[i, j] = self._calculate_iou(boxes[i], boxes[j])
                iou_matrix[j, i] = iou_matrix[i, j]
        
        # 找出需要合并的检测框
        merged_indices = set()
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        for i in range(n):
            if i in merged_indices:
                continue
                
            # 找出与当前框重叠度高的所有框
            overlaps = np.where(iou_matrix[i] > overlap_threshold)[0]
            
            if len(overlaps) <= 1:  # 只有自己
                final_boxes.append(boxes[i])
                final_confidences.append(detections.confidence[i])
                final_class_ids.append(detections.class_id[i])
                continue
            
            # 合并所有重叠框
            merged_indices.update(overlaps)
            
            # 选择置信度最高的框作为最终结果
            best_idx = i
            best_conf = detections.confidence[i]
            
            for j in overlaps:
                if j != i and detections.confidence[j] > best_conf:
                    best_idx = j
                    best_conf = detections.confidence[j]
            
            final_boxes.append(boxes[best_idx])
            final_confidences.append(detections.confidence[best_idx])
            final_class_ids.append(detections.class_id[best_idx])
        
        # 创建新的检测结果
        if not final_boxes:  # 如果没有框，返回原始检测结果
            return detections
            
        # 转换为numpy数组
        final_boxes = np.array(final_boxes)
        final_confidences = np.array(final_confidences)
        final_class_ids = np.array(final_class_ids)
        
        # 创建新的Detections对象
        return sv.Detections(
            xyxy=final_boxes,
            confidence=final_confidences,
            class_id=final_class_ids
        )
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算两个框的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / union if union > 0 else 0
        return iou
    
    def load_image(self, image_source: str) -> np.ndarray:
        """
        加载图像，支持本地路径和URL
        
        参数:
            image_source: 图像路径或URL
            
        返回:
            numpy数组格式的图像
        """
        # 检查是否是URL
        if image_source.startswith(('http://', 'https://')):
            return self.download_image_from_url(image_source)
        
        # 本地文件路径
        if os.path.exists(image_source):
            image = cv2.imread(image_source)
            if image is None:
                try:
                    image = sv.load_image(image_source)
                except Exception as e:
                    raise ValueError(f"无法读取图像文件: {image_source}, 错误: {e}")
            return image
        
        raise FileNotFoundError(f"图像文件不存在: {image_source}")
    
    def download_image_from_url(self, url: str) -> np.ndarray:
        """
        从URL下载图像
        
        参数:
            url: 图像URL
            
        返回:
            numpy数组格式的图像
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 将二进制内容转换为图像
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"无法解码图像: {url}")
                
            return image
        except Exception as e:
            raise ValueError(f"无法从URL下载图像: {url}, 错误: {e}")

# 创建全局检测器实例
_cattle_detector = None

def get_cattle_detector(model_path: str = None, confidence: float = 0.25, iou_threshold: float = 0.5) -> CattleDetector:
    """
    获取牛只检测器实例（单例模式）
    
    参数:
        model_path: YOLO模型路径
        confidence: 检测置信度阈值
        iou_threshold: NMS的IoU阈值
        
    返回:
        CattleDetector实例
    """
    global _cattle_detector
    
    if _cattle_detector is None:
        try:
            _cattle_detector = CattleDetector(model_path, confidence, iou_threshold)
        except ImportError as e:
            logger.error(f"无法创建牛只检测器: {str(e)}")
            raise
    
    return _cattle_detector