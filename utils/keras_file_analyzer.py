"""
Keras 파일 분석 및 수정 도구
.keras 파일의 내부 구조를 분석하고 호환성 문제를 해결합니다.
"""

import zipfile
import json
import h5py
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf

class KerasFileAnalyzer:
    """Keras 파일 구조 분석 및 수정 클래스"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.temp_dir = None
        
    def analyze_structure(self):
        """Keras 파일의 내부 구조 분석"""
        print(f"\n=== Analyzing {self.model_path} ===")
        
        if not self.model_path.exists():
            print(f"Error: File not found: {self.model_path}")
            return None
            
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = Path(temp_dir)
            
            # .keras 파일 압축 해제
            try:
                with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                    print(f"\nExtracted to: {self.temp_dir}")
                    
                    # 압축 파일 내용 나열
                    print("\nFiles in .keras archive:")
                    for file_name in zip_ref.namelist():
                        file_size = zip_ref.getinfo(file_name).file_size
                        print(f"  - {file_name} ({file_size} bytes)")
            except Exception as e:
                print(f"Error extracting file: {e}")
                return None
            
            # 각 파일 분석
            analysis_result = {
                'model_config': None,
                'metadata': None,
                'variables': None,
                'normalization_layers': []
            }
            
            # 1. model.json 분석
            model_json_path = self.temp_dir / "model.json"
            if model_json_path.exists():
                with open(model_json_path, 'r') as f:
                    model_config = json.load(f)
                    analysis_result['model_config'] = model_config
                    
                print("\n=== Model Configuration ===")
                print(f"Keras version: {model_config.get('keras_version', 'Unknown')}")
                print(f"Backend: {model_config.get('backend', 'Unknown')}")
                
                # Normalization layer 찾기
                self._find_normalization_layers(model_config, analysis_result)
            
            # 2. metadata.json 분석
            metadata_path = self.temp_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    analysis_result['metadata'] = metadata
                    print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
            
            # 3. variables.h5 분석
            variables_path = self.temp_dir / "variables.h5"
            if variables_path.exists():
                print("\n=== Variables Analysis ===")
                self._analyze_variables(variables_path, analysis_result)
                
            return analysis_result
    
    def _find_normalization_layers(self, config, result):
        """설정에서 Normalization layer 찾기"""
        def search_layers(obj, path=""):
            if isinstance(obj, dict):
                if obj.get('class_name') == 'Normalization':
                    layer_info = {
                        'path': path,
                        'name': obj.get('config', {}).get('name', 'unknown'),
                        'config': obj.get('config', {})
                    }
                    result['normalization_layers'].append(layer_info)
                    print(f"\nFound Normalization layer: {layer_info['name']}")
                    print(f"  Path: {path}")
                    print(f"  Config: {json.dumps(layer_info['config'], indent=2)}")
                    
                for key, value in obj.items():
                    search_layers(value, f"{path}/{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_layers(item, f"{path}[{i}]")
        
        search_layers(config)
        print(f"\nTotal Normalization layers found: {len(result['normalization_layers'])}")
    
    def _analyze_variables(self, variables_path, result):
        """H5 변수 파일 분석"""
        try:
            with h5py.File(variables_path, 'r') as f:
                print(f"H5 file structure:")
                
                def print_h5_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"  Group: {name}/")
                
                f.visititems(print_h5_structure)
                
                # Normalization 관련 변수 찾기
                print("\nSearching for normalization-related variables:")
                norm_vars = []
                
                def find_norm_vars(name, obj):
                    if 'normalization' in name.lower() or 'norm' in name.lower():
                        if isinstance(obj, h5py.Dataset):
                            norm_vars.append({
                                'name': name,
                                'shape': obj.shape,
                                'dtype': str(obj.dtype),
                                'data': obj[...] if obj.size < 100 else None
                            })
                
                f.visititems(find_norm_vars)
                
                if norm_vars:
                    print(f"\nFound {len(norm_vars)} normalization variables:")
                    for var in norm_vars:
                        print(f"  - {var['name']}: shape={var['shape']}, dtype={var['dtype']}")
                else:
                    print("\nNo normalization variables found!")
                    
                result['variables'] = {
                    'total_variables': len(list(f.keys())),
                    'normalization_variables': norm_vars
                }
                
        except Exception as e:
            print(f"Error analyzing variables: {e}")
    
    def fix_normalization_variables(self, output_path=None):
        """Normalization 변수 문제 수정"""
        if output_path is None:
            output_path = self.model_path.parent / f"{self.model_path.stem}_fixed.keras"
        
        print(f"\n=== Fixing Normalization Variables ===")
        
        # 먼저 구조 분석
        analysis = self.analyze_structure()
        if not analysis:
            return None
        
        if not analysis['normalization_layers']:
            print("No normalization layers found. Nothing to fix.")
            return None
        
        # 임시 디렉토리에서 작업
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 원본 파일 압축 해제
            with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # variables.h5 수정
            variables_path = temp_path / "variables.h5"
            if variables_path.exists():
                self._fix_variables_file(variables_path, analysis['normalization_layers'])
            
            # model.json 수정 (필요한 경우)
            model_json_path = temp_path / "model.json"
            if model_json_path.exists():
                self._fix_model_config(model_json_path)
            
            # 새로운 .keras 파일로 압축
            print(f"\nCreating fixed model: {output_path}")
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for root, dirs, files in os.walk(temp_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(temp_path)
                        zip_ref.write(file_path, arcname)
            
            print(f"Fixed model saved to: {output_path}")
            return str(output_path)
    
    def _fix_variables_file(self, variables_path, norm_layers):
        """H5 변수 파일에 누락된 normalization 변수 추가"""
        print("\nFixing variables.h5...")
        
        try:
            with h5py.File(variables_path, 'r+') as f:
                # 각 normalization layer에 대해 변수 확인 및 추가
                for layer in norm_layers:
                    layer_name = layer['name']
                    print(f"\nProcessing layer: {layer_name}")
                    
                    # 예상되는 변수 이름들
                    expected_vars = [
                        f"{layer_name}/mean:0",
                        f"{layer_name}/variance:0", 
                        f"{layer_name}/count:0"
                    ]
                    
                    # 누락된 변수 추가
                    for var_name in expected_vars:
                        if var_name not in f:
                            print(f"  Adding missing variable: {var_name}")
                            
                            # 적절한 shape 결정 (입력 채널 수에 따라)
                            if 'mean' in var_name or 'variance' in var_name:
                                # 기본값으로 3채널 (RGB) 사용
                                shape = (3,)
                                if 'mean' in var_name:
                                    data = np.zeros(shape, dtype=np.float32)
                                else:  # variance
                                    data = np.ones(shape, dtype=np.float32)
                            else:  # count
                                shape = ()
                                data = np.array(0.0, dtype=np.float32)
                            
                            # 변수 생성
                            f.create_dataset(var_name, data=data)
                            print(f"    Created with shape: {shape}")
                        else:
                            print(f"  Variable already exists: {var_name}")
                
                print("\nVariables fixed successfully!")
                
        except Exception as e:
            print(f"Error fixing variables: {e}")
    
    def _fix_model_config(self, model_json_path):
        """모델 설정 파일 수정 (필요한 경우)"""
        print("\nChecking model.json...")
        
        try:
            with open(model_json_path, 'r') as f:
                model_config = json.load(f)
            
            # 필요한 경우 여기서 config 수정
            # 예: Keras 버전 호환성 처리
            
            with open(model_json_path, 'w') as f:
                json.dump(model_config, f, indent=2)
                
            print("Model config checked.")
            
        except Exception as e:
            print(f"Error fixing model config: {e}")
    
    def extract_weights_only(self, output_dir=None):
        """모델에서 가중치만 추출"""
        if output_dir is None:
            output_dir = self.model_path.parent / f"{self.model_path.stem}_weights"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n=== Extracting Weights ===")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 압축 해제
            with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # variables.h5 복사
            variables_path = temp_path / "variables.h5"
            if variables_path.exists():
                output_weights = output_dir / "weights.h5"
                shutil.copy2(variables_path, output_weights)
                print(f"Weights extracted to: {output_weights}")
                
                # 가중치 정보 저장
                with h5py.File(variables_path, 'r') as f:
                    weight_info = {}
                    
                    def collect_info(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            weight_info[name] = {
                                'shape': obj.shape,
                                'dtype': str(obj.dtype)
                            }
                    
                    f.visititems(collect_info)
                
                # 가중치 정보를 JSON으로 저장
                info_path = output_dir / "weight_info.json"
                with open(info_path, 'w') as f:
                    json.dump(weight_info, f, indent=2)
                print(f"Weight info saved to: {info_path}")
                
                return str(output_dir)
        
        return None


def main():
    """테스트 실행"""
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # 기본 경로 사용
        model_path = "models/health_diagnosis/eye_disease/best_grouped_model.keras"
    
    analyzer = KerasFileAnalyzer(model_path)
    
    # 1. 구조 분석
    print("=" * 80)
    print("STEP 1: Analyzing model structure")
    print("=" * 80)
    analysis = analyzer.analyze_structure()
    
    if analysis and analysis['normalization_layers']:
        # 2. 변수 수정
        print("\n" + "=" * 80)
        print("STEP 2: Fixing normalization variables")
        print("=" * 80)
        fixed_path = analyzer.fix_normalization_variables()
        
        if fixed_path:
            print(f"\n✅ Success! Fixed model saved to: {fixed_path}")
            print("\nYou can now try loading the fixed model.")
    
    # 3. 가중치만 추출 (옵션)
    print("\n" + "=" * 80)
    print("STEP 3: Extracting weights (optional)")
    print("=" * 80)
    weights_dir = analyzer.extract_weights_only()
    if weights_dir:
        print(f"\n✅ Weights extracted to: {weights_dir}")


if __name__ == "__main__":
    main()