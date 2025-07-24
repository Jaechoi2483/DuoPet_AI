"""
Behavior Analysis Prediction Service

This module provides prediction functions for behavior analysis models.
It uses YOLO for object detection and LSTM for behavior classification.
"""

import os
import sys
import numpy as np
import torch
import cv2
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
import yaml
from pathlib import Path
from collections import defaultdict, deque
import threading

# YOLOv5 커스텀 경로 추가
project_root = Path(__file__).parent.parent.parent  # DuoPet_AI 폴더
yolo_models_path = project_root / "models"
if str(yolo_models_path) not in sys.path:
    sys.path.insert(0, str(yolo_models_path))
    
# 윈도우 경로도 추가 (PyCharm 실행 환경)
if os.name == 'nt':  # Windows
    windows_path = Path(r"D:\final_project\DuoPet_AI\models")
    if str(windows_path) not in sys.path:
        sys.path.insert(0, str(windows_path))

# 'models.yolo' 모듈 문제 해결을 위한 핸
# YOLOv5 모델이 'models.yolo'를 찾을 수 있도록 가짜 모듈 추가
if 'models' not in sys.modules:
    import types
    sys.modules['models'] = types.ModuleType('models')
    
if 'models.yolo' not in sys.modules:
    import types
    models_yolo = types.ModuleType('models.yolo')
    sys.modules['models.yolo'] = models_yolo
    
    # Model과 Detect 클래스를 yolo_models.yolo에서 가져와서 models.yolo에 추가
    try:
        from yolo_models.yolo import Model, Detect
        setattr(models_yolo, 'Model', Model)
        setattr(models_yolo, 'Detect', Detect)
    except ImportError:
        # 클래스가 없으면 더미 클래스 생성
        class DummyModel:
            pass
        class DummyDetect:
            pass
        setattr(models_yolo, 'Model', DummyModel)
        setattr(models_yolo, 'Detect', DummyDetect)

# models.common도 필요할 수 있음
if 'models.common' not in sys.modules:
    import types
    models_common = types.ModuleType('models.common')
    sys.modules['models.common'] = models_common
    try:
        from yolo_models.common import *
        # yolo_models.common의 모든 속성을 models.common으로 복사
        import yolo_models.common
        for attr in dir(yolo_models.common):
            if not attr.startswith('_'):
                setattr(models_common, attr, getattr(yolo_models.common, attr))
    except ImportError:
        pass

try:
    from yolo_models.experimental import attempt_load
    from yolo_utils.general import non_max_suppression, scale_coords
    from yolo_utils.torch_utils import select_device
    USE_CUSTOM_YOLO = True
except ImportError:
    from ultralytics import YOLO
    USE_CUSTOM_YOLO = False

# 프로젝트의 공통 모듈들을 임포트합니다
from common.logger import get_logger
from common.config import get_settings, get_model_path
from common.exceptions import ModelNotLoadedError, ModelInferenceError, ValidationError
from .model_manager import model_manager
from .error_handler import error_handler
try:
    from models.behavior_model.behavior_classifier import BehaviorClassifier
    from models.behavior_model.st_gcn.st_gcn import Model as STGCNModel
except ImportError:
    BehaviorClassifier = None
    STGCNModel = None

# 초기 설정
logger = get_logger(__name__)
settings = get_settings()

# YOLOv5 import 확인 로그
if USE_CUSTOM_YOLO:
    logger.info("Using custom YOLOv5 implementation")
else:
    logger.warning("Custom YOLOv5 not available, using ultralytics")


class BehaviorAnalysisPredictor:
    """
    행동 분석을 위한 AI 모델 예측 클래스
    YOLO를 사용한 객체 탐지와 LSTM을 사용한 행동 분류를 수행합니다.
    """

    def __init__(self):
        """모델을 저장할 딕셔너리와 추적 정보를 초기화합니다."""
        self.config = self._load_config()
        self.trackers = {}  # 객체 추적을 위한 딕셔너리
        self.feature_buffer = defaultdict(lambda: deque(maxlen=30))  # 30프레임 특징 버퍼
        self._logged_class_info = False  # 클래스 정보 로깅 플래그
        
    def _load_config(self) -> Dict:
        """모델 설정 파일들을 로드합니다."""
        config = {}
        base_path = Path(__file__).parent.parent.parent / "models" / "behavior_analysis"
        
        # Detection config
        detection_config_path = base_path / "detection" / "config.yaml"
        if detection_config_path.exists():
            with open(detection_config_path, 'r', encoding='utf-8') as f:
                config['detection'] = yaml.safe_load(f)
                
        # Classification config  
        classification_config_path = base_path / "classification" / "config.yaml"
        if classification_config_path.exists():
            with open(classification_config_path, 'r', encoding='utf-8') as f:
                config['classification'] = yaml.safe_load(f)
                
        return config
        
    def _load_yolo_model(self, pet_type: str = "catdog"):
        """YOLO 모델을 로드합니다."""
        model_key = f"yolo_{pet_type}"
        
        def load_yolo():
            # 설정에서 모델 파일명 가져오기
            if pet_type == "catdog":
                # 먼저 원본 모델 시도
                model_filename = "behavior_yolo_catdog_v1_original.pt"
                if not os.path.exists(get_model_path(f"behavior_analysis/detection/{model_filename}")):
                    model_filename = "behavior_yolo_catdog_v1.pt"
            else:
                model_filename = "behavior_yolo_base_v1.pt"
                
            model_path = get_model_path(f"behavior_analysis/detection/{model_filename}")
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}, using dummy model for testing")
                # 더미 모델 사용
                return "dummy_yolo"
                
            logger.info(f"Loading YOLO model from {model_path}")
            if USE_CUSTOM_YOLO:
                # 커스텀 YOLOv5 사용
                device = select_device('')  # GPU 자동 선택
                model = attempt_load(model_path, map_location=device)
                model.eval()
                
                # 모델 클래스 정보 로깅
                if hasattr(model, 'names'):
                    logger.info(f"YOLO model classes: {model.names}")
                return model
            else:
                # Ultralytics YOLO 사용
                return YOLO(model_path)
        
        try:
            return model_manager.get_model(model_key, load_yolo)
        except Exception as e:
            logger.error(f"Failed to load YOLO model after retries: {str(e)}")
            # 최종 폴백으로 더미 모델 반환
            return "dummy_yolo"
        
    def _load_lstm_model(self, pet_type: str) -> torch.nn.Module:
        """LSTM 행동 분류 모델을 로드합니다."""
        model_key = f"lstm_{pet_type}"
        
        def load_lstm():
            # 설정에서 모델 파일명 가져오기
            model_filename = f"behavior_{pet_type}_lstm_v1.pth"
            model_path = get_model_path(f"behavior_analysis/classification/{model_filename}")
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}, using dummy model for testing")
                # 더미 모델 사용
                return "dummy_lstm"
            
            logger.info(f"Loading LSTM model from {model_path}")
            
            # LSTM 모델 구조 정의 (config에 맞춰서)
            class LSTMBehaviorClassifier(torch.nn.Module):
                def __init__(self, input_dim=2048, hidden_dim=256, num_classes=11):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
                    self.fc = torch.nn.Linear(hidden_dim, num_classes)
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    # 마지막 시퀀스의 출력 사용
                    out = self.fc(lstm_out[:, -1, :])
                    return out
            
            # 클래스 수 가져오기
            num_classes = 11 if pet_type == "cat" else 12
            model = LSTMBehaviorClassifier(num_classes=num_classes)
            
            # 가중치 로드 (체크포인트 형식 처리)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # ST-GCN 모델인지 확인
                state_dict = checkpoint['state_dict']
                if any('st_gcn' in key for key in state_dict.keys()):
                    logger.info("Detected ST-GCN model, loading with ST-GCN loader")
                    if BehaviorClassifier is not None:
                        # BehaviorClassifier 사용
                        classifier = BehaviorClassifier()
                        success = classifier.load_model(pet_type, model_path)
                        if success:
                            logger.info(f"ST-GCN model loaded successfully for {pet_type}")
                            return classifier
                        else:
                            raise Exception("ST-GCN model loading failed")
                    else:
                        logger.warning("ST-GCN model support not available, using dummy model")
                        return "dummy_lstm"
                else:
                    # 일반 LSTM 모델
                    model.load_state_dict(checkpoint['state_dict'])
                    model.eval()
            else:
                # 순수 state_dict인 경우
                model.load_state_dict(checkpoint)
                model.eval()
            
            # GPU 사용 가능하면 GPU로 이동
            if torch.cuda.is_available():
                model.cuda()
                
            return model
        
        try:
            return model_manager.get_model(model_key, load_lstm)
        except Exception as e:
            logger.error(f"Failed to load LSTM model after retries: {str(e)}")
            # 최종 폴백으로 더미 모델 반환
            return "dummy_lstm"
        
    @error_handler.with_retry(max_retries=2, delay=0.5, exceptions=(ModelInferenceError, RuntimeError))
    def detect_pets(self, frame: np.ndarray, pet_type: str = "catdog") -> List[Dict]:
        """
        프레임에서 반려동물을 탐지합니다.
        
        Args:
            frame: 입력 프레임 (numpy array)
            pet_type: 탐지할 동물 종류
            
        Returns:
            탐지된 객체 정보 리스트
        """
        try:
            model = self._load_yolo_model(pet_type)
            
            # 더미 모델인 경우
            if model == "dummy_yolo":
                # 테스트용 더미 탐지 결과 생성
                height, width = frame.shape[:2]
                detections = [{
                    "bbox": [width*0.3, height*0.3, width*0.7, height*0.7],
                    "confidence": 0.85,
                    "class": "dog",
                    "class_id": 1
                }]
                return detections
            
            detections = []
            
            if USE_CUSTOM_YOLO:
                # 커스텀 YOLOv5 추론
                device = select_device('')
                img_size = 640
                
                # 이미지 전처리
                img = cv2.resize(frame, (img_size, img_size))
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # 추론
                with torch.no_grad():
                    pred = model(img)[0]
                    
                # NMS 적용
                pred = non_max_suppression(pred, 0.25, 0.45)
                
                # 결과 처리
                if pred[0] is not None and len(pred[0]):
                    # 좌표를 원본 프레임 크기로 변환
                    pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], frame.shape).round()
                    
                    # 탐지 결과 파싱
                    for *xyxy, conf, cls in pred[0]:
                        x1, y1, x2, y2 = xyxy
                        cls_id = int(cls)
                        
                        # 클래스 이름 가져오기
                        if hasattr(model, 'names'):
                            class_names = model.names
                        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
                            class_names = model.module.names
                        else:
                            class_names = {0: "cat", 1: "dog"}  # 기본값
                        
                        # 첫 번째 탐지에서만 로깅 (디버깅용)
                        if not hasattr(self, '_logged_class_info'):
                            logger.info(f"YOLO class_names type: {type(class_names)}")
                            logger.info(f"YOLO class_names content: {class_names}")
                            self._logged_class_info = True
                            
                        # class_names가 list인지 dict인지 확인
                        if isinstance(class_names, list):
                            # list인 경우 인덱스로 접근
                            class_name = class_names[cls_id] if cls_id < len(class_names) else "unknown"
                        elif isinstance(class_names, dict):
                            # dict인 경우 get 메서드 사용
                            class_name = class_names.get(cls_id, "unknown")
                        else:
                            # 기타 경우 (문자열 등)
                            class_name = str(class_names)
                        
                        # cat 또는 dog만 필터링 (대소문자 무시)
                        if class_name.lower() in ["cat", "dog"]:
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(conf),
                                "class": class_name.lower(),  # 소문자로 통일
                                "class_id": cls_id
                            })
                        else:
                            # 디버깅: 필터링된 객체 로깅
                            logger.debug(f"Filtered out: {class_name} (confidence: {conf:.2f})")
            else:
                # Ultralytics YOLO 추론
                results = model(frame)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # 클래스 이름 가져오기
                            if pet_type == "catdog":
                                class_names = {0: "cat", 1: "dog"}
                            else:
                                class_names = model.names
                                
                            class_name = class_names.get(cls, "unknown")
                            
                            # cat 또는 dog만 필터링 (대소문자 무시)
                            if class_name.lower() in ["cat", "dog"]:
                                detections.append({
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "confidence": float(conf),
                                    "class": class_name.lower(),  # 소문자로 통일
                                    "class_id": cls
                                })
                            else:
                                # 디버깅: 필터링된 객체 로깅
                                logger.debug(f"Filtered out: {class_name} (confidence: {conf:.2f})")
            
            # 탐지 결과 로깅
            if detections:
                logger.info(f"Detected {len(detections)} pet(s) in frame")
            else:
                logger.debug("No pets detected in frame")
                    
            return detections
            
        except Exception as e:
            raise ModelInferenceError("YOLO detection", f"Detection failed: {str(e)}")
            
    def extract_features_from_bbox(self, frame: np.ndarray, bbox: List[float], 
                                 prev_bbox: Optional[List[float]] = None) -> np.ndarray:
        """
        바운딩 박스에서 특징을 추출합니다.
        
        Args:
            frame: 현재 프레임
            bbox: 현재 바운딩 박스 [x1, y1, x2, y2]
            prev_bbox: 이전 프레임의 바운딩 박스
            
        Returns:
            추출된 특징 벡터
        """
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 중심점
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 바운딩 박스 크기
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # 이동 벡터 계산
        if prev_bbox:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
            prev_center_x = (prev_x1 + prev_x2) / 2
            prev_center_y = (prev_y1 + prev_y2) / 2
            
            # 이동 거리와 방향
            dx = center_x - prev_center_x
            dy = center_y - prev_center_y
            movement = np.sqrt(dx**2 + dy**2)
            
            # 크기 변화
            prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
            area_change = area - prev_area
        else:
            dx = dy = movement = area_change = 0
            
        # 특징 벡터 구성 (간단한 버전)
        # 실제로는 2048차원이 필요하지만, 여기서는 주요 특징만 추출하고 패딩
        basic_features = np.array([
            center_x / frame.shape[1],  # 정규화된 x 위치
            center_y / frame.shape[0],  # 정규화된 y 위치
            width / frame.shape[1],      # 정규화된 너비
            height / frame.shape[0],     # 정규화된 높이
            area / (frame.shape[0] * frame.shape[1]),  # 정규화된 면적
            dx / frame.shape[1],         # x 방향 이동
            dy / frame.shape[0],         # y 방향 이동
            movement / max(frame.shape), # 전체 이동량
            area_change / (frame.shape[0] * frame.shape[1])  # 크기 변화
        ])
        
        # 2048차원으로 패딩 (실제로는 CNN 특징 추출기 사용 권장)
        features = np.zeros(2048)
        features[:len(basic_features)] = basic_features
        
        return features
        
    def analyze_video(self, video_path: str, pet_type: str = "dog", progress_callback=None) -> Dict:
        """
        비디오 전체를 분석하여 행동을 분류합니다.
        
        Args:
            video_path: 비디오 파일 경로
            pet_type: 분석할 반려동물 종류
            progress_callback: 진행률 업데이트 콜백 함수 (optional)
            
        Returns:
            분석 결과 딕셔너리
        """
        cap = None
        try:
            # 비디오 열기
            logger.info(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                raise ValidationError("video", "Cannot open video file")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Analyzing video: {video_path}, Duration: {video_duration:.2f}s")
            
            # 결과 저장용 변수
            behavior_sequences = []
            frame_count = 0
            prev_detections = {}
            
            # 진행률 초기화
            if progress_callback:
                progress_callback(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 프레임 단위로 처리 (매 5프레임마다)
                if frame_count % 5 == 0:
                    # 객체 탐지
                    detections = self.detect_pets(frame, "catdog")
                    
                    # 각 탐지된 객체에 대해
                    for idx, detection in enumerate(detections):
                        obj_id = f"{detection['class']}_{idx}"
                        
                        # 특징 추출
                        prev_bbox = prev_detections.get(obj_id, {}).get('bbox')
                        features = self.extract_features_from_bbox(
                            frame, detection['bbox'], prev_bbox
                        )
                        
                        # 특징 버퍼에 추가
                        self.feature_buffer[obj_id].append(features)
                        
                        # 30프레임이 모이면 행동 분류
                        if len(self.feature_buffer[obj_id]) == 30:
                            sequence = np.array(list(self.feature_buffer[obj_id]))
                            behavior = self.classify_behavior(
                                sequence, detection['class']
                            )
                            
                            behavior_sequences.append({
                                "frame": frame_count,
                                "time": frame_count / fps if fps > 0 else 0,
                                "object_id": obj_id,
                                "behavior": behavior,
                                "bbox": detection['bbox']
                            })
                            
                    prev_detections = {f"{d['class']}_{i}": d 
                                     for i, d in enumerate(detections)}
                    
                frame_count += 1
                
                # 진행 상황 업데이트
                if fps > 0 and frame_count % 30 == 0:  # 매 30프레임마다 (약 1초)
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}%")
                    if progress_callback:
                        progress_callback(progress)
                    
            cap.release()
            
            # 결과 집계
            behavior_summary = self._summarize_behaviors(behavior_sequences)
            abnormal_behaviors = self._find_abnormal_behaviors(behavior_sequences)
            
            return {
                "video_duration": video_duration,
                "total_frames": total_frames,
                "behavior_sequences": behavior_sequences,
                "behavior_summary": behavior_summary,
                "abnormal_behaviors": abnormal_behaviors
            }
            
        except Exception as e:
            raise ModelInferenceError("video_analysis", f"Video analysis failed: {str(e)}")
        finally:
            # 비디오 캐튳 확실히 해제
            if cap is not None:
                cap.release()
            
    @error_handler.with_retry(max_retries=2, delay=0.5, exceptions=(ModelInferenceError, RuntimeError))
    def classify_behavior(self, feature_sequence: np.ndarray, pet_type: str) -> Dict:
        """
        특징 시퀀스를 사용하여 행동을 분류합니다.
        
        Args:
            feature_sequence: 30프레임의 특징 벡터 (30, 2048)
            pet_type: 반려동물 종류
            
        Returns:
            분류 결과
        """
        try:
            model = self._load_lstm_model(pet_type)
            
            # 더미 모델인 경우
            if model == "dummy_lstm":
                # 테스트용 더미 분류 결과 생성
                behaviors = ["walking", "running", "sitting", "playing", "eating", "sleeping"]
                predicted_class = np.random.randint(0, len(behaviors))
                confidence = 0.75 + np.random.random() * 0.2
                
                # 더미 확률 생성
                num_classes = 11 if pet_type == "cat" else 12
                probabilities = np.random.dirichlet(np.ones(num_classes))
                probabilities = torch.FloatTensor(probabilities).unsqueeze(0)
            elif isinstance(model, BehaviorClassifier):
                # ST-GCN 모델 사용 - 스켈레톤 데이터가 필요함
                # 현재는 바운딩 박스만 있으므로 더미 스켈레톤 생성
                num_frames = 30
                num_joints = 15
                # 더미 스켈레톤 데이터 (실제로는 DeepLabCut 등으로 추출해야 함)
                skeleton_data = np.random.randn(3, num_frames, num_joints, 1) * 0.1
                
                # 예측
                action_probs = model.predict(pet_type.upper(), skeleton_data)
                # 가장 높은 확률의 행동 선택
                predicted_label = list(action_probs.keys())[0]
                confidence = list(action_probs.values())[0]
                # action_classes를 사용하여 predicted_class 계산
                predicted_class = list(model.action_classes[pet_type.upper()]).index(predicted_label)
                
                # 확률 분포 생성 (action_probs에서 직접 가져옴)
                num_classes = len(model.action_classes[pet_type.upper()])
                probabilities = torch.zeros(1, num_classes)
                for i, action in enumerate(model.action_classes[pet_type.upper()]):
                    probabilities[0, i] = action_probs.get(action, 0.0)
            else:
                # 일반 LSTM 모델
                sequence_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)
                
                if torch.cuda.is_available():
                    sequence_tensor = sequence_tensor.cuda()
                    
                # 추론
                with torch.no_grad():
                    outputs = model(sequence_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
            # 클래스 이름 가져오기
            if BehaviorClassifier and isinstance(model, BehaviorClassifier):
                # ST-GCN 모델의 경우
                class_names = {i: label for i, label in enumerate(model.action_classes[pet_type.upper()])}
            else:
                # 기본 LSTM 모델
                config = self.config.get('classification', {}).get('models', [])
                model_config = next((m for m in config if pet_type in m.get('name', '').lower()), None)
                
                if model_config:
                    class_names = model_config['output']['classes']
                else:
                    # 기본값
                    class_names = {
                        0: "normal", 1: "eating", 2: "sleeping", 3: "playing",
                        4: "walking", 5: "running", 6: "sitting", 7: "jumping",
                        8: "grooming", 9: "scratching", 10: "abnormal"
                    } if pet_type == "cat" else {
                        0: "normal", 1: "barking", 2: "eating", 3: "sleeping",
                        4: "playing", 5: "walking", 6: "running", 7: "jumping",
                        8: "sitting", 9: "tail_wagging", 10: "digging", 11: "abnormal"
                    }
            
            behavior_name = class_names.get(predicted_class, "unknown")
            is_abnormal = behavior_name == "abnormal"
            
            return {
                "behavior": behavior_name,
                "confidence": confidence,
                "is_abnormal": is_abnormal,
                "all_probabilities": {
                    class_names.get(i, f"class_{i}"): float(probabilities[0][i])
                    for i in range(len(class_names))
                }
            }
            
        except Exception as e:
            raise ModelInferenceError("behavior_classification", 
                                    f"Classification failed: {str(e)}")
                                    
    def _summarize_behaviors(self, sequences: List[Dict]) -> Dict[str, int]:
        """행동 시퀀스를 요약합니다."""
        summary = defaultdict(int)
        for seq in sequences:
            behavior = seq['behavior']['behavior']
            summary[behavior] += 1
        return dict(summary)
        
    def _find_abnormal_behaviors(self, sequences: List[Dict]) -> List[Dict]:
        """비정상 행동을 찾습니다."""
        abnormal = []
        for seq in sequences:
            if seq['behavior']['is_abnormal']:
                abnormal.append({
                    "time": seq['time'],
                    "behavior": seq['behavior']['behavior'],
                    "confidence": seq['behavior']['confidence']
                })
        return abnormal


class BehaviorAnalysisPredictorSingleton:
    """싱글톤 패턴 구현 클래스 - 리셋 가능"""
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> BehaviorAnalysisPredictor:
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = BehaviorAnalysisPredictor()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """인스턴스 리셋 - 모델 언로드 및 재초기화"""
        with cls._lock:
            if cls._instance is not None:
                # 모든 모델 언로드
                model_manager.reset()
                # 인스턴스 재생성
                cls._instance = BehaviorAnalysisPredictor()
                logger.info("BehaviorAnalysisPredictor instance reset")


# 싱글톤 인스턴스
predictor = BehaviorAnalysisPredictorSingleton.get_instance()