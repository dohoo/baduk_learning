import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import tkinter as tk
from tkinter import messagebox
import threading

# --- 상수 정의 ---
BOARD_SIZE = 19
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE
NUM_CHANNELS = 3 
STONE_RADIUS = 15 # 바둑돌 크기
GRID_SIZE = 30    # 격자 간격
BOARD_PADDING = 30 # 바둑판 테두리 여백
KOMI = 6.5 # 백돌(AI)의 덤
# AI 기권 임계값 (백돌 승률이 이 값보다 낮으면 기권합니다)
RESIGN_THRESHOLD = 0.05 

# --- 1. 모델 로드 ---
MODEL_PATH = 'go_policy_network_supervised.h5'
policy_model = None

if not os.path.exists(MODEL_PATH):
    print(f"🚨 오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
    print("AI를 실행하려면 학습된 모델 파일(.h5)이 스크립트와 같은 경로에 있어야 합니다.")
    print("더미 모델로 대체합니다. AI가 무작위 수만 둘 수 있습니다.")
    policy_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS)),
        keras.layers.Dense(NUM_ACTIONS, activation='softmax')
    ])
else:
    try:
        # TensorFlow 로그를 최소화
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        policy_model = keras.models.load_model(MODEL_PATH)
        print(f"✅ 정책 모델 '{MODEL_PATH}' 로드 성공.")
    except Exception as e:
        print(f"🚨 모델 로드 중 오류 발생: {e}")
        policy_model = keras.Sequential([
            keras.layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS)),
            keras.layers.Dense(NUM_ACTIONS, activation='softmax')
        ])

# --- 2. GoBoard 클래스 정의 (게임 로직) ---
class GoBoard:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        # 0: 빈 칸, 1: 흑돌 (Black, User), 2: 백돌 (White, AI)
        self.board = np.zeros((size, size), dtype=np.int32)
        self.current_player = 1  
        self.is_game_over = False
        self.pass_count = 0
        self.winner = None # 'B' (Black) or 'W' (White)
        # NOTE: 코 규칙(Ko rule)은 구현되지 않았습니다.

    def get_neighbors(self, r, c):
        """ 주어진 좌표의 상하좌우 이웃 좌표를 반환합니다. """
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def get_group_liberties(self, r, c):
        """ 
        (r, c)에 있는 돌의 연결 그룹과 활로(Liberties)를 BFS를 사용하여 찾습니다.
        반환: (group: 돌 그룹 좌표 집합, liberty_count: 활로 개수)
        """
        if not (0 <= r < self.size and 0 <= c < self.size) or self.board[r, c] == 0:
            return set(), 0

        color = self.board[r, c]
        group = set()
        liberties = set()
        q = [(r, c)]
        
        while q:
            curr_r, curr_c = q.pop(0)
            if (curr_r, curr_c) in group:
                continue
            
            group.add((curr_r, curr_c))
            
            for nr, nc in self.get_neighbors(curr_r, curr_c):
                neighbor_stone = self.board[nr, nc]
                if neighbor_stone == 0:
                    liberties.add((nr, nc))
                elif neighbor_stone == color and (nr, nc) not in group:
                    q.append((nr, nc))
        
        return group, len(liberties)

    def remove_stones(self, group):
        """ 주어진 그룹의 돌들을 보드에서 제거하고 제거된 돌의 수를 반환합니다. """
        count = len(group)
        for r, c in group:
            self.board[r, c] = 0
        return count

    def is_valid_move(self, r, c):
        """ 착수가 유효한지 확인합니다. (경계, 빈 칸, 자살 수) """
        if self.is_game_over:
            return False
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.board[r, c] != 0:
            return False
        
        # 1. 임시로 돌을 놓습니다.
        player = self.current_player
        self.board[r, c] = player
        
        # 2. 따낸 돌이 있는지 확인 (자살 방지 1)
        # 돌을 놓은 후 주변의 상대편 그룹을 확인하여 따냄이 발생하는지 봅니다.
        captured_stones = 0
        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr, nc] == 3 - player: # 상대 돌
                group, liberties = self.get_group_liberties(nr, nc)
                if liberties == 0:
                    captured_stones += len(group)
        
        # 3. 자살 수 확인 (자살 방지 2)
        # 따냄이 없었는데, 놓은 돌 그룹의 활로가 0개인지 확인합니다.
        if captured_stones == 0:
            group, liberties = self.get_group_liberties(r, c)
            if liberties == 0:
                # 자살 수이므로 원래대로 되돌립니다.
                self.board[r, c] = 0
                return False
                
        # 4. 임시로 놓았던 돌을 다시 비웁니다.
        self.board[r, c] = 0
        
        # NOTE: 이 함수는 따냄을 임시로 검사하는 용도이며, 실제 돌 제거는 make_move에서 합니다.
        return True
        
    def make_move(self, r, c):
        """ 돌을 놓고 턴을 넘기며, 따냄과 자살 수 규칙을 적용합니다. """
        if not self.is_valid_move(r, c):
            return False
            
        player = self.current_player
        
        # 1. 돌을 놓습니다.
        self.board[r, c] = player
        
        # 2. 상대 돌 제거 (따냄)
        captured_stones = 0
        groups_to_remove = []
        
        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr, nc] == 3 - player:
                group, liberties = self.get_group_liberties(nr, nc)
                if liberties == 0:
                    groups_to_remove.append(group)
        
        for group in groups_to_remove:
            captured_stones += self.remove_stones(group)
        
        # 3. 턴 전환 및 패스 카운트 초기화
        self.current_player = 3 - self.current_player  # Switch 1 -> 2, 2 -> 1
        self.pass_count = 0
        
        return True
        
    def pass_turn(self):
        """ 패스 처리 및 게임 종료 확인 """
        self.current_player = 3 - self.current_player
        self.pass_count += 1
        if self.pass_count >= 2:
            self.is_game_over = True
        
    def resign(self):
        """ 기권 처리 및 승자 결정 """
        if self.is_game_over:
            return
            
        self.is_game_over = True
        # 현재 턴인 플레이어가 기권했으므로, 상대 플레이어가 승리합니다.
        self.winner = 'W' if self.current_player == 1 else 'B'
        
    def calculate_score(self):
        """
        단순 지역 (Territory) 계산을 통해 점수를 계산합니다.
        (죽은 돌은 고려하지 않고, 살아있는 돌 + 확보된 지역으로 계산)
        """
        scores = {1: 0, 2: 0} # 1: 흑, 2: 백
        visited_territory = set()

        # 1. 돌 수 계산 (Area-like)
        scores[1] += np.sum(self.board == 1)
        scores[2] += np.sum(self.board == 2)
        
        # 2. 지역 (Territory) 계산
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0 and (r, c) not in visited_territory:
                    
                    territory_group = set()
                    borders = set()
                    q = [(r, c)]
                    
                    # BFS로 빈 공간 그룹 찾기
                    while q:
                        curr_r, curr_c = q.pop(0)
                        if (curr_r, curr_c) in territory_group:
                            continue
                        
                        territory_group.add((curr_r, curr_c))
                        
                        for nr, nc in self.get_neighbors(curr_r, curr_c):
                            neighbor_stone = self.board[nr, nc]
                            if neighbor_stone == 0 and (nr, nc) not in territory_group and (nr, nc) not in q:
                                q.append((nr, nc))
                            elif neighbor_stone != 0:
                                borders.add(neighbor_stone) # 주변을 둘러싼 돌의 색깔 기록
                                
                    # 경계가 한 가지 색깔로만 이루어져 있다면, 그 색깔의 집으로 판정
                    if len(borders) == 1:
                        owner = borders.pop()
                        scores[owner] += len(territory_group)
                    
                    visited_territory.update(territory_group)

        # 3. 백돌에게 덤 적용
        scores[2] += KOMI
        
        return scores

    def get_win_probability(self):
        """
        단순 점수차를 기반으로 흑돌의 승률을 추정합니다.
        (경고: 이는 실제 AI의 승률 예측이 아니며, 단순 점수차에 기반한 근사치입니다.)
        """
        scores = self.calculate_score()
        black_score = scores[1]
        white_score = scores[2]
        score_diff = black_score - white_score
        
        # 점수차를 승률로 변환하는 단순한 휴리스틱
        # (예: 30집 차이가 나면 승률 100% / 0%로 가정하고 선형 보간합니다.)
        
        # 선형 보간 후 0.01과 0.99 사이로 클램핑 (Clamp)
        win_prob_black = 0.5 + (score_diff / 30.0)
        win_prob_black = max(0.01, min(0.99, win_prob_black)) 

        return win_prob_black

    def get_state(self):
        """ Policy Network의 입력 형태 (1, 19, 19, 3)으로 변환합니다. """
        black_stones = (self.board == 1).astype(np.float32)
        white_stones = (self.board == 2).astype(np.float32)
        
        # Current player (1.0 for Black, 0.0 for White)
        player_color = np.full((self.size, self.size), 
                               1.0 if self.current_player == 1 else 0.0, 
                               dtype=np.float32)

        state = np.stack([black_stones, white_stones, player_color], axis=-1)
        return np.expand_dims(state, axis=0)


# --- 3. AI 플레이어 로직 함수 ---
def ai_move(board: GoBoard, model: keras.Model):
    """ AI Player (White) selects and makes a valid move based on policy network prediction. """
    if board.current_player != 2 or board.is_game_over:
        return False
        
    # 1. 승률 확인 및 기권 결정 (AI = 백돌)
    win_prob_black = board.get_win_probability()
    win_prob_white = 1.0 - win_prob_black
    
    # AI 기권 임계값 (상수 RESIGN_THRESHOLD 사용)
    if win_prob_white < RESIGN_THRESHOLD:
        print(f"🤖 AI(백돌) 승률이 {win_prob_white*100:.1f}%로 매우 낮아 기권합니다.")
        board.resign()
        return True # AI가 기권하는 것으로 '착수'를 완료함

    state_input = board.get_state()
    predictions = model.predict(state_input, verbose=0)[0] 
    sorted_indices = np.argsort(-predictions)
    
    best_r, best_c = -1, -1
    found_valid_move = False
    
    # 4. Iterate through sorted moves to find the first valid one
    for action_index in sorted_indices:
        r = action_index // BOARD_SIZE
        c = action_index % BOARD_SIZE
            
        if board.is_valid_move(r, c):
            best_r, best_c = r, c
            found_valid_move = True
            break
            
    if found_valid_move:
        # 5. Make the move
        board.make_move(best_r, best_c)
        move_coord = f"{chr(ord('A') + best_c)}{best_r + 1}"
        print(f"🤖 AI(백돌) 착수: {move_coord}")
        return True
    else:
        # Pass when no valid move found
        print("AI 착수 실패: 바둑판에 더 이상 둘 곳이 없습니다. AI가 패스합니다.")
        board.pass_turn()
        return True 

# --- 4. GUI 클래스 정의 (바둑판 수정 포함) ---
class GoGUI:
    def __init__(self, master):
        self.master = master
        master.title("바둑 AI (정책망)")

        self.game = GoBoard()
        self.policy_model = policy_model 
        
        # Calculate canvas size
        canvas_width = (BOARD_SIZE - 1) * GRID_SIZE + 2 * BOARD_PADDING
        canvas_height = (BOARD_SIZE - 1) * GRID_SIZE + 2 * BOARD_PADDING
        
        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg="#D2B48C") 
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.handle_click) 

        # Status Label
        # 상태 레이블은 이제 2줄로 승률 정보를 표시합니다.
        self.status_label = tk.Label(master, text="게임을 시작합니다. 사용자(흑돌) 차례.", font=('Arial', 12), justify=tk.LEFT)
        self.status_label.pack(pady=5)
        
        # 버튼 프레임 (Pass, Resign, Score Check를 나란히 배치하기 위해)
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=5)
        
        # Pass Button
        self.pass_button = tk.Button(self.button_frame, text="Pass", command=self.handle_pass, font=('Arial', 12), bg='lightgray')
        self.pass_button.pack(side=tk.LEFT, padx=5) # 왼쪽에 배치
        
        # Resign Button (기권 버튼)
        self.resign_button = tk.Button(self.button_frame, text="Resign (기권)", command=self.handle_resign, font=('Arial', 12), bg='#FF6347', fg='white')
        self.resign_button.pack(side=tk.LEFT, padx=5) # 왼쪽에 배치

        # Score Check Button (형세 판단 버튼 추가)
        self.score_check_button = tk.Button(self.button_frame, text="형세 판단 (집 계산)", command=self.handle_score_check, font=('Arial', 12), bg='#4CAF50', fg='white')
        self.score_check_button.pack(side=tk.LEFT, padx=5) # 왼쪽에 배치

        # AI Lock to prevent multiple AI moves simultaneously
        self.ai_lock = threading.Lock()
        
        self.draw_board()
        self.update_status()

    def get_canvas_coord(self, r, c):
        """ Converts (row, column) index to canvas coordinates (x, y) """
        x = c * GRID_SIZE + BOARD_PADDING
        y = r * GRID_SIZE + BOARD_PADDING
        return x, y
        
    def draw_board(self):
        self.canvas.delete("all")
        
        # Actual board boundary coordinates
        start_coord = BOARD_PADDING
        end_coord = (BOARD_SIZE - 1) * GRID_SIZE + BOARD_PADDING

        # 1. Draw board lines and coordinate labels
        for i in range(BOARD_SIZE):
            x, y = self.get_canvas_coord(i, i)

            # Vertical lines
            self.canvas.create_line(x, start_coord, x, end_coord, fill="black")
            
            # Horizontal lines
            self.canvas.create_line(start_coord, y, end_coord, y, fill="black")
            
            # Row labels (1-19) - Y-axis
            self.canvas.create_text(BOARD_PADDING / 2, y, text=str(i + 1), fill="black")
            # Column labels (A-S) - X-axis
            self.canvas.create_text(x, BOARD_PADDING / 2, text=chr(ord('A') + i), fill="black")

        # 2. Draw Star Points (Hoshis)
        star_indices = []
        if BOARD_SIZE == 19:
            star_indices = [4, 10, 16] 
            
        for r_idx in star_indices:
            r = r_idx - 1 # 1-indexed to 0-indexed
            for c_idx in star_indices:
                c = c_idx - 1 # 1-indexed to 0-indexed
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    x, y = self.get_canvas_coord(r, c)
                    
                    # Center point (Tengen) check (10-10 line or 9,9 index)
                    if BOARD_SIZE == 19 and r == 9 and c == 9:
                         self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black")
                    else:
                         self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")

        # 3. Draw Stones
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                player = self.game.board[r, c]
                if player != 0:
                    x, y = self.get_canvas_coord(r, c)
                    color = "black" if player == 1 else "white"
                    outline_color = "black"
                    self.canvas.create_oval(x - STONE_RADIUS, y - STONE_RADIUS, 
                                            x + STONE_RADIUS, y + STONE_RADIUS, 
                                            fill=color, outline=outline_color)

    def update_status(self):
        """ 상태 레이블 업데이트 및 게임 종료 처리 """
        if self.game.is_game_over:
            self.pass_button.config(state=tk.DISABLED)
            self.resign_button.config(state=tk.DISABLED)
            self.score_check_button.config(state=tk.DISABLED) # 게임 종료 시 비활성화
            
            if self.game.winner:
                # 기권으로 인한 종료
                winner_name = "사용자(흑돌)" if self.game.winner == 'B' else "AI(백돌)"
                loser_name = "사용자(흑돌)" if self.game.winner == 'W' else "AI(백돌)"
                
                # AI가 기권했을 때 메시지 수정
                if self.game.current_player == 2 and self.game.winner == 'B':
                    text = (f"게임 종료! AI(백돌) 기권.\n"
                            f"승자: 사용자(흑돌)")
                else:
                    text = (f"게임 종료! {loser_name} 기권.\n"
                            f"승자: {winner_name}")
            else:
                # 2회 패스로 인한 점수 계산 종료
                scores = self.game.calculate_score()
                black_score = scores[1]
                white_score = scores[2]
                
                if black_score > white_score:
                    winner = "사용자(흑돌) 승리!"
                elif white_score > black_score:
                    winner = "AI(백돌) 승리!"
                else:
                    winner = "무승부"
                    
                text = (f"게임 종료! {winner}\n"
                        f"흑돌 점수: {black_score:.1f}, 백돌 점수 (덤 {KOMI} 포함): {white_score:.1f}\n"
                        f"두 번 연속 패스로 인해 종료되었습니다.")
            
        else:
            # --- 승률 계산 및 표시 로직 ---
            win_prob_black = self.game.get_win_probability()
            win_rate_black = f"{win_prob_black * 100:.1f}%"
            win_rate_white = f"{(1 - win_prob_black) * 100:.1f}%"
            
            player_name = '사용자(흑돌, X)' if self.game.current_player == 1 else 'AI(백돌, O)'
            text = (
                f"현재 차례: {player_name} (연속 패스 {self.game.pass_count}/2)\n"
                f"⚫ 흑돌 승률: {win_rate_black} | ⚪ 백돌 승률: {win_rate_white}"
            )
            
        self.status_label.config(text=text)


    def handle_click(self, event):
        """ 사용자의 마우스 클릭을 처리합니다. """
        if self.game.current_player != 1 or self.game.is_game_over or self.ai_lock.locked():
            return 

        # 캔버스 좌표 -> (r, c) 인덱스로 변환
        c = round((event.x - BOARD_PADDING) / GRID_SIZE)
        r = round((event.y - BOARD_PADDING) / GRID_SIZE)
        
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self.game.make_move(r, c):
                self.draw_board()
                self.update_status()
                
                # AI 턴을 별도 스레드에서 시작하여 GUI 응답성 유지
                threading.Thread(target=self.start_ai_play).start()
            else:
                messagebox.showerror("오류", "유효하지 않은 착수입니다. (이미 돌이 있거나 자살 수)")
        else:
            if self.game.current_player == 1:
                messagebox.showinfo("정보", "바둑판 격자점을 클릭하세요.")


    def handle_pass(self):
        """ 사용자가 패스 버튼을 눌렀을 때 처리합니다. """
        if self.game.is_game_over or self.ai_lock.locked():
            return
            
        if self.game.current_player == 1:
            self.game.pass_turn() # 패스 처리 및 턴 전환
            
            self.update_status()
            messagebox.showinfo("패스", "사용자가 패스했습니다. AI 차례입니다.")
            
            if self.game.is_game_over:
                self.update_status()
                return
                
            # AI 턴을 별도 스레드에서 시작
            threading.Thread(target=self.start_ai_play).start()
            
        else:
            messagebox.showwarning("경고", "AI의 차례에는 패스할 수 없습니다.")

    def handle_resign(self):
        """ 사용자가 기권 버튼을 눌렀을 때 처리합니다. """
        if self.game.is_game_over or self.ai_lock.locked():
            return

        if messagebox.askyesno("기권 확인", "정말로 기권하시겠습니까? 기권하면 상대방이 승리합니다."):
            self.game.resign()
            self.update_status()
            messagebox.showinfo("기권", "사용자가 기권했습니다. AI(백돌) 승리!")

    def handle_score_check(self):
        """ 현재 시점의 단순 형세 판단 결과를 보여줍니다. """
        if self.game.is_game_over or self.ai_lock.locked():
            messagebox.showwarning("경고", "게임이 종료되었거나 AI가 계산 중입니다.")
            return
            
        scores = self.game.calculate_score()
        black_score = scores[1]
        white_score = scores[2]
        
        # 현재 코드의 형세 판단은 '살아있는 돌'을 고려하지 않고 단순 지역만 계산합니다.
        
        result_message = (
            f"--- 현재 형세 판단 (단순 지역 계산) ---\n\n"
            f"⚫ 사용자(흑돌) 점수: {black_score:.1f} 점\n"
            f"⚪ AI(백돌) 점수 (덤 {KOMI}점 포함): {white_score:.1f} 점\n\n"
        )
        
        score_diff = black_score - white_score
        
        if score_diff > 0:
            result_message += f"현재 흑돌이 {score_diff:.1f}집 앞서고 있습니다."
        elif score_diff < 0:
            result_message += f"현재 백돌이 {-score_diff:.1f}집 앞서고 있습니다."
        else:
            result_message += "현재 동점입니다."
            
        result_message += "\n\n(참고: 이 계산은 죽은 돌을 고려하지 않은 단순 지역 점수입니다.)"
        
        messagebox.showinfo("형세 판단 결과", result_message)
            
            
    def start_ai_play(self):
        """ AI 로직을 실행하고 GUI를 안전하게 업데이트합니다. """
        if self.game.current_player == 2 and not self.game.is_game_over:
            with self.ai_lock:
                # 1. AI Logic
                ai_move(self.game, self.policy_model)
                
                # 2. GUI 업데이트를 메인 스레드에서 예약
                self.master.after(1, self.update_gui_after_ai)

    def update_gui_after_ai(self):
        """ AI 스레드가 끝난 후 GUI 요소를 안전하게 업데이트합니다. """
        self.draw_board()
        self.update_status()


# --- 5. 프로그램 진입점 ---
if __name__ == "__main__":
    # Tkinter 루트 윈도우 생성
    root = tk.Tk()
    
    app = GoGUI(root)
    
    # 이벤트 루프 시작
    root.mainloop()