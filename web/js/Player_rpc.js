// Agents that represent either a player or an AI
function Player(color) {
    this.color = color;
}

Player.prototype.myTurn = function () {
    this.game.setCurrentColor(this.color);
    gameInfo.setText((function (string) {
            return string.charAt(0).toUpperCase() + string.slice(1);
        })(this.color) + "'s turn.");
    gameInfo.setColor(this.color);
    gameInfo.setBlinking(false);
};

Player.prototype.watch = function () {
};

Player.prototype.setGo = function (r, c) {
    return this.game.setGo(r, c, this.color);
};

function HumanPlayer(color, game) {
    Player.call(this, color, game);
}

HumanPlayer.prototype = new Player();

HumanPlayer.prototype.myTurn = function () {
    Player.prototype.myTurn.call(this);
    this.game.toHuman(this.color);
    if (this.other instanceof AIPlayer) {
        gameInfo.setText('Your turn');
    }
};

function AIPlayer(mode, color, game) {
    Player.call(this, color, game);
    this.computing = false;
    this.cancel = 0;
    this.mode = mode;
    this.server = 'http://dpl04.skyeyes.lycc.qihoo.net:2221/action?model=policy_dl';
    //this.server = 'http://dpl04.skyeyes.lycc.qihoo.net:2222/action?model=policy_rl';
    //this.server = 'http://dpl04.skyeyes.lycc.qihoo.net:2223/action?model=policy_rollout';
    var self = this;
}

AIPlayer.prototype = new Player();

AIPlayer.prototype.myTurn = function () {
    Player.prototype.myTurn.call(this);
    this.game.toOthers();
    gameInfo.setText("Thinking...");
    gameInfo.setBlinking(true);
    this.move();
};

AIPlayer.prototype.watch = function (r, c, color) {

    //this.worker.postMessage({
    //    type: 'watch',
    //    r: r,
    //    c: c,
    //    color: color
    //});
};

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}



AIPlayer.prototype.move = function () {
    if (this.game.rounds === 0) {
        this.setGo(7, 7);
    } else if (this.game.rounds === 1) {
        var moves = [
            [6, 6],
            [6, 7],
            [6, 8],
            [7, 6],
            [7, 7],
            [7, 8],
            [8, 6],
            [8, 7],
            [8, 8]
        ];
        while (true) {
            var ind = Math.floor(Math.random() * moves.length);
            if (this.setGo(moves[ind][0], moves[ind][1])) {
                return;
            } else {
                moves.splice(ind, 1);
            }
        }
    } else {
        var board_stream = this.game.get_board();
        var ai_player = this;
        ai_player.computing = true;
        $.ajax({
            type: "POST",
            url: this.server,
            contentType: "application/json",
            data: JSON.stringify({type: "compute", board: board_stream, player: this.color}),
            success: function (data) {
                var ret = data;
                while(true) {
                    var action = indexOfMax(ret["action"]);
                    var r = Math.floor(action / 15);
                    var c = action % 15;
                    var is_ok = ai_player.setGo(r, c);
                    if(is_ok) {
                        break;
                    } else {
                        ret["action"][action] = 0;
                    }
                }
                ai_player.computing = false;
            },
            failure: function (errMsg) {
                alert(errMsg);
            }
        });

    }
};

