#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import re
import random
from math import ceil
import numpy as np
from bs4 import BeautifulSoup
from sklearn.externals.joblib import Parallel, delayed

from util.AI_db import DBWrapper
from util.AI_logger import logger
from AI_renju import RenjuGame

stream_reg = re.compile("(?:[01]{15}\|){14}[01]{15}")


def is_legal_stream(stream):
    return stream_reg.match(stream) is not None


def board_to_stream(board):
    """
    :param board: a numpy array (15 x 15)
    :return: a string stream
    """
    return '|'.join(map(lambda x: ''.join(map(str, x)), board))


def stream_to_board(stream):
    """
    :param stream: a string stream
    :return: a numpy array (15 x 15)
    """
    return np.array(map(lambda x: map(np.int8, x), stream.split('|')))


class RenjuPatterns(object):
    def __init__(self):
        self.db = DBWrapper(db_path="./data/patterns.db")
        self.db.execute("CREATE TABLE IF NOT EXISTS pattern(id INTEGER PRIMARY KEY AUTOINCREMENT,  \
                                                          pattern TEXT NOT NULL, \
                                                          player SMALLINT, \
                                                          action SMALLINT)")
        self.ids = map(lambda row: row["id"], self.db.query("select id from pattern"))
        self.fetch_index = 0

    def import_RenjuPattern(self, _corpus, batch_size=128):
        for idx, samples in enumerate(_corpus.iterator_fetch_rows(batch_size)):
            for sample in samples:
                patterns = ','.join(map(str, sample[0].get_patterns()))
                player = sample[0].get_player()
                action = sample[1]
                self.db.execute("insert INTO pattern(pattern, player, action) VALUES (?, ?, ?)",
                                patterns, player, action)
            if idx % 10 == 0:
                print "handle step=", idx

    def shuffle_datas(self):
        random.shuffle(self.ids)

    def num_batchs_per_epochs(self, batch_size):
        return ceil(len(self.ids) / float(batch_size))

    def iterator_fetch_rows(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        for offset in range(0, len(self.ids), batch_size):
            limit_no = min(len(self.ids), offset + batch_size)
            batch_ids = ','.join(map(str, self.ids[offset: limit_no]))
            rows = self.db.query("SELECT pattern,action FROM pattern WHERE id in (%s)" % batch_ids)
            samples = []
            for row in rows:
                pattern = map(int, row["pattern"].split(','))
                action = row["action"]
                samples.append((pattern, action))
            yield samples

    def next_fetch_rows(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        start_idx = self.fetch_index
        end_idx = min(self.fetch_index + batch_size, len(self.ids))
        if start_idx >= end_idx:
            self.fetch_index = 0
            self.shuffle_datas()
            start_idx, end_idx = 0, batch_size
        batch_ids = ','.join(map(str, self.ids[start_idx: end_idx]))
        rows = self.db.query("SELECT pattern,action FROM pattern WHERE id in (%s)" % batch_ids)
        samples = []
        for row in rows:
            pattern = map(int, row["pattern"].split(','))
            action = row["action"]
            samples.append((pattern, action))
        self.fetch_index = end_idx
        return samples


class RenjuCorpus(object):
    def __init__(self):
        self.db = DBWrapper(db_path="./data/renju.db")
        self.db.execute("CREATE TABLE IF NOT EXISTS renju(id INTEGER PRIMARY KEY AUTOINCREMENT,  \
                                                          gid INTEGER NOT NULL, \
                                                          mid SMALLINT NOT NULL, \
                                                          board TEXT NOT NULL, \
                                                          player SMALLINT, \
                                                          action SMALLINT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS renju_gid ON renju(gid)")
        self.ids = map(lambda row: row["id"], self.db.query("select id from renju"))
        self.fetch_index = 0

    def import_RenjuNet(self, file_path):
        if not os.path.exists(file_path):
            logger.error("not found file: %s" % file_path, to_exit=True)
        # read xml file
        bs_tree = BeautifulSoup(open(file_path, 'r').read())
        games = bs_tree.find_all("game")
        # insert moves
        game_num = len(games)
        move_count = 0
        step = 0
        for game in games:
            step += 1
            gid = int(game.attrs["id"])
            moves = game.move.text.strip().replace("%20", " ").split(" ")
            if len(self.db.query("select id from renju WHERE gid=?", gid)) > 0:  # when gid exists
                continue
            renju_game = RenjuGame()
            for mid, move in enumerate(moves):
                move = move.strip()
                if move == "":
                    continue
                board_stream = board_to_stream(renju_game.board)
                player = renju_game.player
                row = ord(move[0]) - ord('a')
                col = int(move[1:]) - 1
                action = renju_game.transform_action((row, col))
                # insert
                self.db.execute("insert INTO renju (gid, mid, board, player, action) VALUES (?, ?, ?, ?, ?)",
                                gid, mid, board_stream, player, action)
                # do move
                renju_game.do_move((row, col))
            move_count += len(moves)
            if step % 100 == 0:
                print "load games= %d / %d" % (step, game_num)
        logger.info("newly insert games=%d, moves=%d" % (game_num, move_count))
        print "finish import moves"

    def random_fetch_rows(self, fetch_size):
        """
        :param fetch_size:
        :return: a list of tuples (instance of RenjuGame, action of int)
        """
        ids = random.sample(self.ids, fetch_size)
        # rows = self.db.query("SELECT board,player,action FROM renju ORDER BY RANDOM() LIMIT ?", fetch_size)
        rows = self.db.query("SELECT board,player,action FROM renju where id IN (%s)" % ",".join(map(str, ids)))
        samples = []
        for row in rows:
            board = stream_to_board(row["board"])
            player = row["player"]
            action = row["action"]
            samples.append((RenjuGame(board=board, player=player), action))
        return samples

    def shuffle_datas(self):
        random.shuffle(self.ids)

    def num_batchs_per_epochs(self, batch_size):
        return int(ceil(len(self.ids) / float(batch_size)))

    def num_samples(self):
        return len(self.ids)

    def iterator_fetch_rows(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        for offset in range(0, len(self.ids), batch_size):
            limit_no = min(len(self.ids), offset + batch_size)
            batch_ids = ','.join(map(str, self.ids[offset: limit_no]))
            rows = self.db.query("SELECT board,player,action FROM renju WHERE id in (%s)" % batch_ids)
            samples = []
            for row in rows:
                board = stream_to_board(row["board"])
                player = row["player"]
                action = row["action"]
                samples.append((RenjuGame(board=board, player=player), action))
            while len(samples) < batch_size:
                samples.append(random.choice(samples))
            yield samples

    def next_fetch_rows(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        start_idx = self.fetch_index
        end_idx = min(self.fetch_index + batch_size, len(self.ids))
        if start_idx >= end_idx:
            self.fetch_index = 0
            self.shuffle_datas()
            start_idx, end_idx = 0, batch_size
        batch_ids = ','.join(map(str, self.ids[start_idx: end_idx]))
        rows = self.db.query("SELECT board,player,action FROM renju WHERE id in (%s)" % batch_ids)
        samples = []
        for row in rows:
            board = stream_to_board(row["board"])
            player = row["player"]
            action = row["action"]
            samples.append((RenjuGame(board=board, player=player), action))
        self.fetch_index = end_idx
        return samples

corpus = RenjuCorpus()
patterns = RenjuPatterns()


def parallel_import_renju_pattern(n_jobs=20):
    Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(single_import_renju_pattern)(index) for index in xrange(len(corpus.ids)))


def single_import_renju_pattern(index):
    renju_db = DBWrapper(db_path="./data/renju.db")
    pattern_db = DBWrapper(db_path="./data/patterns.db")
    row = renju_db.query("select * from renju limit 1 offset ?", index)[0]
    position = RenjuGame(board=stream_to_board(row["board"]), player=row["player"])
    pattern = ','.join(map(str, position.get_patterns()))
    action = row["action"]
    while True:
        try:
            pattern_db.execute("insert INTO pattern(pattern, player, action) VALUES (?, ?, ?)",
                                pattern, row["player"], action)
            break
        except:
            logger.warn("fail to insert into pattern_db, try again")


if __name__ == "__main__":
    # patterns.import_RenjuPattern(corpus)
    parallel_import_renju_pattern(n_jobs=32)
