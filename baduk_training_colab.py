# 바둑 AI 학습 스크립트 (Google Colab 환경)
# 구글 드라이브 연결 → SGF 파일 수집 → 배치 학습 + 체크포인트 + 8배 데이터증강

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gzip
import shutil
from pathlib import Path
import json
from typing import List, Tuple
import pickle
from datetime import datetime

# ============================================================================
# 1. Google Colab 환경 설정
# ============================================================================

def setup_google_drive():
    """Google Drive 연결"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        drive_root = '/content/drive/MyDrive'
        print(f"✅ Google Drive 연결 성공: {drive_root}")
        return drive_root
    except ImportError:
        print("⚠️  Google Colab 환경이 아닙니다. 로컬 경로 사용")
        return './data'

# ============================================================================
# 2. SGF 파일 수집 및 분석
# ============================================================================

def collect_sgf_files(drive_root: str) -> List[Path]:
    """
    baduk/Pro/1/, baduk/Pro/2/, ... 등에서 모든 .sgf 파일 수집
    """
    sgf_files = []
    base_path = Path(drive_root) / 'baduk' / 'Pro'
    
    if not base_path.exists():
        print(f"⚠️  경로가 없습니다: {base_path}")
        return []
    
    # N은 자연수 (1, 2, 3, ...)
    for n_folder in sorted(base_path.iterdir()):
        if n_folder.is_dir() and n_folder.name.isdigit():
            sgf_files.extend(n_folder.glob('*.sgf'))
    
    print(f"✅ 발견된 SGF 파일: {len(sgf_files)}개")
    return sorted(sgf_files)

def ask_training_count(total_files: int) -> int:
    """학습할 파일 수를 사용자에게 물어봄"""
    print(f"\n총 {total_files}개의 SGF 파일을 찾았습니다.")
    print(f"학습에 사용할 파일 수를 입력하세요 (1-{total_files}):")
    
    while True:
        try:
            count = int(input())
            if 1 <= count <= total_files:
                return count
            else:
                print(f"1-{total_files} 사이의 값을 입력하세요.")
        except ValueError:
            print("정수를 입력하세요.")

# ============================================================================
# 3. SGF 파싱 함수
# ============================================================================

def parse_sgf(sgf_path: Path) -> List[Tuple[np.ndarray, int]]:
    """
    SGF 파일을 파싱하여 (board_state, move) 쌍 리스트 반환
    
    Returns:
        List[(board_state: (19, 19, 3), move: 0-360)]
    """
    try:
        with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 간단한 SGF 파싱 (실제로는 더 복잡할 수 있음)
        training_data = []
        
        # SGF에서 move sequence 추출
        move_list = extract_moves_from_sgf(content)
        
        if not move_list:
            return []
        
        # Board 재현 및 training data 생성
        board = np.zeros((19, 19, 3), dtype=np.float32)
        for move_idx, (row, col, color) in enumerate(move_list):
            if 0 <= row < 19 and 0 <= col < 19:
                # color: 0 = black, 1 = white
                board[row, col, color] = 1
                move_label = row * 19 + col
                training_data.append((board.copy(), move_label, color))
        
        return training_data
    
    except Exception as e:
        print(f"❌ SGF 파싱 실패 ({sgf_path.name}): {e}")
        return []

def extract_moves_from_sgf(content: str) -> List[Tuple[int, int, int]]:
    """
    SGF 내용에서 움직임 추출
    반환: [(row, col, color), ...] where color is 0 (black) or 1 (white)
    """
    moves = []
    import re
    
    # Black moves: ;B[xx]
    black_pattern = r';B\[([a-s]{2})\]'
    white_pattern = r';W\[([a-s]{2})\]'
    
    # 순서대로 모든 move 찾기
    game_pattern = r'\(.*?\)'
    games = re.findall(game_pattern, content, re.DOTALL)
    
    if not games:
        return []
    
    game = games[0]
    move_sequence = re.findall(r';([BW])\[([a-s]{2})\]', game)
    
    for color_char, coords in move_sequence:
        try:
            col = ord(coords[0]) - ord('a')
            row = ord(coords[1]) - ord('a')
            color = 0 if color_char == 'B' else 1
            moves.append((row, col, color))
        except:
            continue
    
    return moves

# ============================================================================
# 4. 데이터 증강 (8배)
# ============================================================================

def augment_data(board: np.ndarray, move: int) -> List[Tuple[np.ndarray, int]]:
    """
    바둑판의 회전, 반전을 이용하여 8배 데이터 증강
    
    Returns:
        8개의 (augmented_board, augmented_move) 쌍
    """
    augmented = []
    
    def transform_move(move, transform_type):
        """move 번호를 변환"""
        row, col = move // 19, move % 19
        
        if transform_type == 0:  # original
            return move
        elif transform_type == 1:  # 90도 회전
            new_row, new_col = col, 18 - row
        elif transform_type == 2:  # 180도 회전
            new_row, new_col = 18 - row, 18 - col
        elif transform_type == 3:  # 270도 회전
            new_row, new_col = 18 - col, row
        elif transform_type == 4:  # 수평 반전
            new_row, new_col = row, 18 - col
        elif transform_type == 5:  # 수평 반전 + 90도
            new_row, new_col = 18 - col, 18 - row
        elif transform_type == 6:  # 수평 반전 + 180도
            new_row, new_col = 18 - row, col
        elif transform_type == 7:  # 수평 반전 + 270도
            new_row, new_col = col, row
        
        return new_row * 19 + new_col
    
    for i in range(8):
        if i == 0:  # original
            aug_board = board.copy()
        elif i == 1:  # 90도 회전
            aug_board = np.rot90(board, 1)
        elif i == 2:  # 180도 회전
            aug_board = np.rot90(board, 2)
        elif i == 3:  # 270도 회전
            aug_board = np.rot90(board, 3)
        elif i == 4:  # 수평 반전
            aug_board = np.fliplr(board)
        elif i == 5:  # 수평 반전 + 90도
            aug_board = np.rot90(np.fliplr(board), 1)
        elif i == 6:  # 수평 반전 + 180도
            aug_board = np.rot90(np.fliplr(board), 2)
        elif i == 7:  # 수평 반전 + 270도
            aug_board = np.rot90(np.fliplr(board), 3)
        
        aug_move = transform_move(move, i)
        augmented.append((aug_board, aug_move))
    
    return augmented

# ============================================================================
# 5. 모델 정의
# ============================================================================

def create_policy_network(input_shape=(19, 19, 3)):
    """바둑 정책 네트워크 생성"""
    model = keras.Sequential([
        layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(361, activation='softmax')  # 19x19 = 361
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# 6. 학습 데이터 파이프라인 (스트리밍 처리)
# ============================================================================

class StreamingTrainingPipeline:
    """파싱 → 증강 → 학습 → 삭제를 연속적으로 처리"""
    
    def __init__(self, sgf_files: List[Path], batch_size: int = 32, augment_factor: int = 8):
        self.sgf_files = sgf_files
        self.batch_size = batch_size
        self.augment_factor = augment_factor
        self.data_buffer = []
        self.labels_buffer = []
    
    def process_batch(self, sgf_files_batch: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치의 SGF 파일들을 파싱 → 증강하여 학습 데이터 생성
        """
        self.data_buffer = []
        self.labels_buffer = []
        
        for sgf_file in sgf_files_batch:
            print(f"  파싱 중: {sgf_file.name}")
            training_data = parse_sgf(sgf_file)
            
            for board, move, color in training_data:
                # 8배 증강
                augmented_samples = augment_data(board, move)
                for aug_board, aug_move in augmented_samples:
                    self.data_buffer.append(aug_board)
                    self.labels_buffer.append(aug_move)
        
        # 메모리에 로드
        X = np.array(self.data_buffer, dtype=np.float32)
        y = np.array(self.labels_buffer, dtype=np.int32)
        
        print(f"  생성된 학습 데이터: {len(X)}개 샘플")
        
        # 메모리 정리
        self.data_buffer = []
        self.labels_buffer = []
        
        return X, y

# ============================================================================
# 7. 체크포인트 및 로깅
# ============================================================================

class TrainingLogger:
    """학습 진행 상황 기록"""
    
    def __init__(self, log_dir: str = './training_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.logs = {
            'start_time': datetime.now().isoformat(),
            'batches': []
        }
    
    def log_batch(self, batch_num: int, loss: float, accuracy: float, samples: int):
        """배치 결과 기록"""
        self.logs['batches'].append({
            'batch': batch_num,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'samples': samples,
            'timestamp': datetime.now().isoformat()
        })
        self._save()
    
    def _save(self):
        """로그 저장"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)

# ============================================================================
# 8. 메인 학습 루프
# ============================================================================

def train_baduk_ai(drive_root: str, num_epochs: int = 3, batch_sgf_size: int = 300):
    """
    메인 학습 함수
    
    Args:
        drive_root: Google Drive 루트 경로
        num_epochs: 에포크 수
        batch_sgf_size: 한 번에 학습할 SGF 파일 수
    """
    
    # 1. SGF 파일 수집
    print("\n" + "="*60)
    print("🔍 단계 1: SGF 파일 수집")
    print("="*60)
    
    sgf_files = collect_sgf_files(drive_root)
    
    if not sgf_files:
        print("❌ SGF 파일을 찾을 수 없습니다.")
        return
    
    # 2. 학습할 파일 수 결정
    print("\n" + "="*60)
    print("❓ 단계 2: 학습 데이터 크기 결정")
    print("="*60)
    
    num_files = ask_training_count(len(sgf_files))
    sgf_files_to_train = sgf_files[:num_files]
    
    print(f"\n📊 학습 계획:")
    print(f"  - 총 SGF 파일: {num_files}개")
    print(f"  - 배치 크기: {batch_sgf_size}개 파일")
    print(f"  - 배치 수: {(num_files + batch_sgf_size - 1) // batch_sgf_size}개")
    print(f"  - 데이터 증강: 8배")
    
    # 3. 모델 생성
    print("\n" + "="*60)
    print("🤖 단계 3: 모델 생성")
    print("="*60)
    
    model_dir = Path(drive_root) / 'baduk_models'
    model_dir.mkdir(exist_ok=True)
    
    model = create_policy_network()
    print("✅ 정책 네트워크 생성 완료")
    print(model.summary())
    
    # 4. 파이프라인 및 로거 초기화
    pipeline = StreamingTrainingPipeline(sgf_files_to_train, batch_size=32, augment_factor=8)
    logger = TrainingLogger(str(model_dir / 'logs'))
    
    # 5. 배치 학습
    print("\n" + "="*60)
    print("🚀 단계 4: 배치 학습 시작")
    print("="*60)
    
    total_batches = (num_files + batch_sgf_size - 1) // batch_sgf_size
    
    for batch_idx in range(total_batches):
        print(f"\n📚 배치 {batch_idx + 1}/{total_batches}")
        print("-" * 60)
        
        # 배치 범위
        start_idx = batch_idx * batch_sgf_size
        end_idx = min((batch_idx + 1) * batch_sgf_size, num_files)
        batch_files = sgf_files_to_train[start_idx:end_idx]
        
        # 데이터 파이프라인 실행
        X_batch, y_batch = pipeline.process_batch(batch_files)
        
        if len(X_batch) == 0:
            print(f"⚠️  배치 {batch_idx + 1}에서 데이터를 생성하지 못했습니다. 스킵합니다.")
            continue
        
        # 모델 학습
        print(f"\n🔥 학습 중... ({len(X_batch)}개 샘플)")
        history = model.fit(
            X_batch, y_batch,
            epochs=num_epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # 로그 저장
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        logger.log_batch(batch_idx + 1, final_loss, final_acc, len(X_batch))
        
        # 체크포인트 저장
        checkpoint_path = model_dir / f'checkpoint_batch_{batch_idx + 1:03d}.h5'
        model.save(str(checkpoint_path))
        print(f"💾 체크포인트 저장: {checkpoint_path.name}")
        
        # 메모리 정리
        del X_batch, y_batch
        import gc
        gc.collect()
        
        print(f"✅ 배치 {batch_idx + 1} 완료 (Loss: {final_loss:.4f}, Acc: {final_acc:.4f})")
    
    # 6. 최종 모델 저장
    print("\n" + "="*60)
    print("🎉 학습 완료!")
    print("="*60)
    
    final_model_path = model_dir / 'baduk_ai_final.h5'
    model.save(str(final_model_path))
    print(f"✅ 최종 모델 저장: {final_model_path}")
    
    # 학습 요약 출력
    print(f"\n📊 학습 요약:")
    print(f"  - 총 SGF 파일 처리: {num_files}개")
    print(f"  - 총 배치 수: {total_batches}개")
    print(f"  - 최종 모델 경로: {final_model_path}")
    print(f"  - 체크포인트 경로: {model_dir}")
    print(f"  - 로그 경로: {logger.log_file}")

# ============================================================================
# 9. 실행
# ============================================================================

print("\n🏴 바둑 AI 학습 스크립트 시작")
print("=" * 60)

# Google Drive 연결
drive_root = setup_google_drive()

# 학습 실행
train_baduk_ai(
    drive_root=drive_root,
    num_epochs=3,           # 에포크
    batch_sgf_size=300      # 배치당 SGF 파일 수
)

print("\n" + "="*60)
print("✅ 전체 프로세스 완료!")
print("="*60)
