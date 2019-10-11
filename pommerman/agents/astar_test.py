from astar import astar

board = [[0,1,0],[0,2,0],[0,1,0]]

res = astar(board, (0,0), (2,2), [0,2])
print(res)