import time
import math
import random
import numpy as np
from helper import *
from typing import Optional, Set, Tuple, List


class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game

        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = "ai"
        self.player_string = "Player {}: ai".format(player_number)
        self.timer = timer
        self.C = 1.5
        self.p = 1  # Initial probability for making a frame move
        self.total_moves = 0  # Track the total number of moves made

    def get_move(self, state: np.array) -> Tuple[int, int]:
        valid_moves = get_valid_actions(state)

        for move in valid_moves:
            me_win, _ = check_win(state, move, self.player_number)
            opp_win, _ = check_win(state, move, 3 - self.player_number)

            if me_win:
                return tuple(map(int, move))
            elif opp_win:
                return tuple(map(int, move))

        center = (state.shape[0] // 2, state.shape[1] // 2)

        if state[center] == 0:
            return tuple(map(int, center))

        self.update_probability(state=state)

        frames = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == self.player_number:
                    frames.extend(self.get_frame_cells(i, j, state))

        frames = list(set(frames))

        if random.random() < self.p and frames:
            selected_frame = random.choice(frames)
            return tuple(map(int, selected_frame))

        move = self.mcts(state)
        return tuple(map(int, move))

    def get_frame_cells(self, i: int, j: int, state: np.array) -> List[Tuple[int, int]]:
        frames = [
            (i - 1, j - 2),
            (i - 1, j + 2),
            (i + 1, j - 2),
            (i + 1, j + 2),
            (i - 2, j - 1),
            (i - 2, j + 1),
            (i + 2, j - 1),
            (i + 2, j + 1),
            (i, j),
        ]
        return [
            frame
            for frame in frames
            if 0 <= frame[0] < state.shape[0]
            and 0 <= frame[1] < state.shape[1]
            and state[frame] == 0
        ]

    def update_probability(self, state: np.array):

        self.total_moves += 1
        board_size = state.shape[0]
        decay_factor = 0

        if board_size == 4:
            decay_factor = 0.87
        elif board_size == 6:
            decay_factor = 0.96

        # small_board_decay = 0.9500
        # large_board_decay = 0.9999

        # min_board_size = 4
        # max_board_size = 40
        # decay_factor = small_board_decay + (large_board_decay - small_board_decay) * (
        #     (board_size - min_board_size) / (max_board_size - min_board_size)
        # )

        # decay_factor = max(
        #     small_board_decay, min(large_board_decay, decay_factor)
        # )  # Clamp to range

        self.p = max(0.01, self.p * decay_factor)  # Apply decay factor

    def mcts_iterations(self, state: np.array) -> int:
        time_sec = fetch_remaining_time(self.timer, self.player_number)
        board_size = state.shape[0]
        remaining_moves = np.count_nonzero(state == 0)
        base_iterations = 1050
        if board_size == 6:
            base_iterations = 5000
        time_factor = int(200 * time_sec)
        if board_size == 6:
            time_factor = int(400 * time_sec)
        move_factor = int(remaining_moves * 10)  # Adjust the multiplier as needed
        if board_size == 6:
            move_factor = int(remaining_moves * 15)
        return max(base_iterations, min(time_factor, move_factor))

    def mcts(self, state: np.array) -> Tuple[int, int]:
        root = Node(state, None, None)
        self.visits = {root: 0}
        self.wins = {root: 0}
        iterations = self.mcts_iterations(state)
        # print(f"Iterations: {iterations}")
        for _ in range(iterations):  # Number of iterations
            node = self.select(root)
            if node is None:
                continue
            self.expand(node)
            result = self.simulate(node)
            self.backpropagate(node, result)
        if not root.children:
            return random.choice(
                get_valid_actions(state)
            )  # Fallback to a random move if no children
        return max(root.children, key=lambda n: self.visits.get(n, 0)).move

    def select(self, node):
        # UCT selection strategy
        beta = 0.5  # Define beta with an appropriate value
        best_value = -float("inf")
        best_node = None
        for child in node.children:  # Iterate directly over the list
            if child not in self.visits:
                return child
            beta = 500 / (500 + self.visits[child])
            uct_value = (self.wins[child] / self.visits[child]) + self.C * np.sqrt(
                np.log(self.visits[node]) / self.visits[child]
            )
            rave_value = child.rave_wins / (child.rave_visits + 1)
            combined_value = (
                beta * uct_value + (1 - beta) * rave_value
            )  # Use uct_value here
            if combined_value > best_value:
                best_value = combined_value
                best_node = child
        return best_node

    def expand(self, node):
        valid_moves = get_valid_actions(node.state)
        for move in valid_moves:
            new_state = node.state.copy()
            new_state[move] = self.player_number
            new_node = Node(new_state, move, node)
            node.children.append(new_node)

    def simulate(self, node):
        current_state = node.state.copy()
        current_player = self.player_number
        while True:
            valid_moves = get_valid_actions(current_state)
            if not valid_moves:
                return 0  # Draw
            move = random.choice(valid_moves)
            current_state[move] = current_player
            if check_win(current_state, move, current_player):
                return 1 if current_player == self.player_number else -1
            current_player = 3 - current_player

    def backpropagate(self, node, result):
        while node is not None:
            if node not in self.visits:
                self.visits[node] = 0
                self.wins[node] = 0
            self.visits[node] += 1
            self.wins[node] += result
            node = node.parent
            result = -result


class Node:
    def __init__(self, state, move, parent):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.rave_wins = 0  # RAVE wins
        self.rave_visits = 0  # RAVE visits

    def __hash__(self):
        return hash(str(self.state.tostring()) + str(self.move))

    def __eq__(self, other):
        return (
            isinstance(other, Node)
            and np.array_equal(self.state, other.state)
            and self.move == other.move
        )
