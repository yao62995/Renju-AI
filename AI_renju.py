#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import random
import numpy as np


class RenjuGame(object):
    BOARD_SIZE = 15
    PLAYER_BLACK, PLAYER_WHITE = 1, 2
    STONE_EMPTY, STONE_BLACK, STONE_WHITE = 0, 1, 2
    DATA_TYPE = np.int8

    def __init__(self, board=None, player=None):
        if board is None:
            self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=self.DATA_TYPE)
        else:
            self.board = board
        if player is None:
            self.player = self.PLAYER_BLACK
        else:
            self.set_player(player)

    def set_player(self, player):
        if type(player) is str:
            if player == "black":
                self.player = RenjuGame.PLAYER_BLACK
            else:
                self.player = RenjuGame.PLAYER_WHITE
        else:
            self.player = player

    def get_board(self):
        return self.board

    def get_player(self):
        return self.player

    def get_player_name(self):
        return "black" if self.player == RenjuGame.PLAYER_BLACK else "white"

    def is_legal_move(self, move):
        return self.board[move] == self.STONE_EMPTY

    def switch_player(self):
        if self.player == self.PLAYER_BLACK:
            self.player = self.PLAYER_WHITE
        else:
            self.player = self.PLAYER_BLACK

    def do_move(self, move):
        self.board[move] = self.player
        self.switch_player()

    def reset_game(self):
        RenjuGame.__init__(self)

    def game_over(self, move):
        for direct in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            found = 1
            for d in [-1, 1]:
                for i in range(1, 5):
                    next = (move[0] + direct[0] * i * d, move[1] + direct[1] * i * d)
                    if (next[0] < 0 or next[0] >= self.BOARD_SIZE
                        or next[1] < 0 or next[1] >= self.BOARD_SIZE
                        or self.board[next] != self.player):
                        break
                    else:
                        found += 1
            if found >= 5:
                return True
        return False

    def legal_actions(self):
        """
        :return: list of legal board position, such as POS(1,1)
        """
        actions = []
        for row in xrange(self.BOARD_SIZE):
            for col in xrange(self.BOARD_SIZE):
                if self.board[row, col] == self.STONE_EMPTY:
                    actions.append(row * self.BOARD_SIZE + col)
        return actions

    def legal_action(self, action):
        move = transform_action(action)
        if self.board[move] == RenjuGame.STONE_EMPTY:
            return True
        return False

    def choose_action(self, predict_vals):
        sorted_actions = np.argsort(predict_vals)[::-1]
        for action in sorted_actions:
            if self.legal_action(action):
                return action
        return None

    def weighted_choose_action(self, predict_vals):
        min_probs, max_probs = min(predict_vals), max(predict_vals)
        predict_vals = map(lambda prob: (prob - min_probs) / (max_probs - min_probs), predict_vals)
        # total = sum(predict_vals)
        # predict_vals = map(lambda prob: prob / total, predict_vals)
        r = random.uniform(0, sum(predict_vals))
        upto = 0
        for idx, prob in enumerate(predict_vals):
            if upto + prob >= r and self.legal_action(idx):
                return idx
            upto += prob
        return None

    def random_action(self):
        ok_pos = np.where(self.board == RenjuGame.STONE_EMPTY)
        if len(ok_pos[0]) == 0:
            return None
        pos = random.randint(0, len(ok_pos[0]) - 1)
        return ok_pos[0][pos] * RenjuGame.BOARD_SIZE + ok_pos[1][pos]

    def step_games(self, action, player_plane=False):
        """
        :param action: pos (int)
        :return: game state
        """
        if action is None:
            return self.get_states(player_plane=player_plane)
        move = transform_action(action)
        terminal = self.game_over(move)
        reward = 0
        if terminal:
            reward = 1 if self.player == self.PLAYER_BLACK else -1
        self.do_move(move)
        state = self.get_states(player_plane=player_plane)
        return state, reward, terminal

    def read_games(self, game_moves):
        """
        :param game_movesgame_moves: ex: [(8, 8), (8, 9)]
        :return:
        """
        states = []
        actions = []
        for action in game_moves:
            state = self.get_states()
            self.do_move(action)
            states.append(state)
            actions.append(action)
        return states, actions

    # def forward(self, action, copy=True):
    #     if copy:
    #         game = RenjuGame()
    #         game.board = self.board
    #         game.player = self.player
    #     move = transform_action(action)

    def get_states(self, player_plane=False, flatten=False):
        """
        :param recorded_game: a instance of RenjuGame
        :param player_plane: whether add current player plane to feature planes
        :return:
        """
        state = self.feature_stone_color()
        if player_plane:
            feat = self.feature_player_plane()
            state = np.append(state, feat, axis=2)
        if flatten:
            state = state.flatten('F')
        return state

    def replicate_game(self):
        game = RenjuGame()
        game.board = np.copy(self.board)
        game.player = self.player
        return game

    def feature_stone_color(self):
        feature = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 3), dtype=self.DATA_TYPE)
        if self.player == RenjuGame.PLAYER_BLACK:
            op_color = RenjuGame.PLAYER_WHITE
        else:
            op_color = RenjuGame.PLAYER_BLACK
        stone_color_dict = {self.player: 0, op_color: 1, RenjuGame.STONE_EMPTY: 2}
        for row in xrange(self.BOARD_SIZE):
            for col in xrange(self.BOARD_SIZE):
                stone_status = self.board[row, col]
                feature[row, col, stone_color_dict[stone_status]] = 1
        return feature

    def feature_player_plane(self):
        if self.player == self.PLAYER_BLACK:
            return np.ones((self.BOARD_SIZE, self.BOARD_SIZE, 1), dtype=self.DATA_TYPE)
        else:
            return np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, 1), dtype=self.DATA_TYPE)

    def get_patterns(self):
        return get_renju_pattern_features(self)


def transform_action(action):
    """
    :param action:  transform (1, 2) => (1*15+2)    OR    (1*15+2) => (1, 2)
    :return:
    """
    if type(action) == tuple:
        action = action[0] * RenjuGame.BOARD_SIZE + action[1]
    else:
        action = (action / RenjuGame.BOARD_SIZE, action % RenjuGame.BOARD_SIZE)
    return action


def one_hot_action(action):
    """
    :param action: int type
    :return:
    """
    one_hot_act = np.zeros(RenjuGame.BOARD_SIZE * RenjuGame.BOARD_SIZE, dtype=RenjuGame.DATA_TYPE)
    one_hot_act[action] = 1
    return one_hot_act


def find_2d_occurences(array, sub_array):
    count = 0
    w, h = sub_array.shape
    for i in xrange(array.shape[0] - w + 1):
        for j in xrange(array.shape[1] - h + 1):
            if np.array_equal(array[i: (i + w), j: (j + h)], sub_array):
                count += 1
    return count


def find_top_left_diag_occurrences(array, diag):
    count = 0
    n = diag.shape[0]
    for i in xrange(array.shape[0] - n + 1):
        for j in xrange(array.shape[1] - n + 1):
            equal = True
            for k in xrange(n):
                if array[i + k, j + k] != diag[k, k]:
                    equal = False
                    break
            if equal:
                count += 1
    return count


def find_top_right_diag_occurrences(array, diag):
    count = 0
    n = diag.shape[0]
    for i in xrange(array.shape[0] - n + 1):
        for j in xrange(array.shape[1] - n + 1):
            equal = True
            for k in xrange(n - 1, -1, -1):
                if array[i + k, j + n - 1 - k] != diag[k, n - 1 - k]:
                    equal = False
                    break
            if equal:
                count += 1
    return count


def renju_patterns():
    base_pattern = [
        lambda x: [0, x, x, x, x, 0],
        lambda x: [0, x, 0, 0, x, 0],
        lambda x: [x, x, x, x, 0],
        lambda x: [x, 0, 0, 0, x],
        lambda x: [0, 0, x, x, x, 0],
        lambda x: [0, x, 0, 0, 0, 0],
        lambda x: [x, x, 0, 0, x],
        lambda x: [x, x, 0, 0, 0],
    ]
    # pattern_list = dict()
    # pattern_list["horizontal"] = [lambda x: np.array([pattern(x)]) for pattern in base_pattern]
    # pattern_list["vertical"] = [lambda x: np.array([pattern(x)]).T for pattern in base_pattern]
    # pattern_list["diag_top_left"] = [lambda x: np.diag(pattern(x)) for pattern in base_pattern]
    # pattern_list["diag_top_right"] = [lambda x: np.fliplr(np.diag(pattern(x))) for pattern in base_pattern]
    # pattern_black_list = dict((key, [it(RenjuGame.PLAYER_BLACK) for it in item]) for key, item in pattern_list.items())
    # pattern_white_list = dict((key, [it(RenjuGame.PLAYER_WHITE) for it in item]) for key, item in pattern_list.items())
    pattern_black_list, pattern_white_list = dict(), dict()
    pattern_black_list["horizontal"] = [np.array([pattern(RenjuGame.PLAYER_BLACK)]) for pattern in base_pattern]
    pattern_black_list["vertical"] = [np.array([pattern(RenjuGame.PLAYER_BLACK)]).T for pattern in base_pattern]
    pattern_black_list["diag_top_left"] = [np.diag(pattern(RenjuGame.PLAYER_BLACK)) for pattern in base_pattern]
    pattern_black_list["diag_top_right"] = [np.fliplr(np.diag(pattern(RenjuGame.PLAYER_BLACK))) for pattern in
                                            base_pattern]
    pattern_white_list["horizontal"] = [np.array([pattern(RenjuGame.PLAYER_WHITE)]) for pattern in base_pattern]
    pattern_white_list["vertical"] = [np.array([pattern(RenjuGame.PLAYER_WHITE)]).T for pattern in base_pattern]
    pattern_white_list["diag_top_left"] = [np.diag(pattern(RenjuGame.PLAYER_WHITE)) for pattern in base_pattern]
    pattern_white_list["diag_top_right"] = [np.fliplr(np.diag(pattern(RenjuGame.PLAYER_WHITE))) for pattern in
                                            base_pattern]
    return pattern_black_list, pattern_white_list


def get_renju_pattern_features(position):
    """
    :param position: an instance of RenjuGame
    :param pattern_list: variable generated from func<renju_patterns>
    :return: pattern features: list
    """
    board = position.board
    if position.player == RenjuGame.PLAYER_BLACK:
        pattern_list = (PATTERN_LIST[0], PATTERN_LIST[1])
    else:
        pattern_list = (PATTERN_LIST[1], PATTERN_LIST[0])
    pattern_features = []
    for _patterns in pattern_list:
        pattern_horizontal_features = \
            map(lambda sub_arr: find_2d_occurences(board, sub_arr), _patterns['horizontal'])
        pattern_vertical_features = \
            map(lambda sub_arr: find_2d_occurences(board, sub_arr), _patterns["vertical"])
        pattern_diag_tl_features = \
            map(lambda sub_arr: find_top_left_diag_occurrences(board, sub_arr), _patterns["diag_top_left"])
        pattern_diag_tr_features = \
            map(lambda sub_arr: find_top_right_diag_occurrences(board, sub_arr), _patterns["diag_top_right"])

        pattern_features.extend(pattern_horizontal_features)
        pattern_features.extend(pattern_vertical_features)
        pattern_features.extend(pattern_diag_tl_features)
        pattern_features.extend(pattern_diag_tr_features)
    return pattern_features


# global variable
PATTERN_LIST = renju_patterns()

if __name__ == "__main__":
    from AI_import import stream_to_board
    import time

    board = "000000000000000|" \
            "000000000000000|" \
            "000000000000000|" \
            "000000000000000|" \
            "000000000000000|" \
            "000000010000000|" \
            "000000102010000|" \
            "000000212200000|" \
            "000000012000000|" \
            "000000021112000|" \
            "000000000100000|" \
            "000000000020000|" \
            "000000000000000|" \
            "000000000000000|" \
            "000000000000000"
    board = stream_to_board(board)
    player = 2
    position = RenjuGame(board=board, player=player)
    elapsed_time = 0
    for _ in xrange(10):
        start_time = time.time()
        patterns = get_renju_pattern_features(position)
        elapsed_time += int((time.time() - start_time) * 1000)
    print "time: %d (ms)" % elapsed_time
    print len(patterns), "\t", patterns
