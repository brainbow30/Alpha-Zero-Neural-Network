# Alpha Zero Neural Network - 77% First Class Mark
A flask restful API to allow a series of games to ask for predictions on the current game state which the NN will provide through the flask interface so that the game can make an informed decision on the next move. The neural network uses the policy and value approach of Alpha Zero and trains as such. This can be used to train on various games.

Has outperformed myself at games of Othello and Connect-4 with 5000 and 2000 training games respectively.

run "pip install -r requirements.txt" to install all packages needed for program

The final paper is included with all the details of solution, experimentation and results.

## Results
### Othello:
Defeated Monte Carlo tree search with 3x more “thinking time” <br />
Defeated myself with 2s “thinking time”<br />
#### GUI
![Othello GUI](othello.png?raw=true "Othello GUI")
#### AlphaZero vs MCTS
AlphaZero 500ms "thinking time" vs Monte Carlo Tree Search with varying "thinking time"
![AlphaZero (500ms) vs MCTS Results"](EvalOthello.png?raw=true "AlphaZero (500ms) vs MCTS")

### Connect-4:
Defeated Monte Carlo tree search with 4x more “thinking time”<br />
Defeated myself with 500ms “thinking time”<br />
#### GUI
![Connect-4 GUI](connect4GUI.png?raw=true "Connect-4 GUI")

### Checkers:
Defeated Monte Carlo tree search with 5x more “thinking time”<br />
#### GUI
![Checkers GUI](checkers.png?raw=true "Checkers GUI")<br/>





