from tile_coding import *

iht = IHT(800)

tiles_vector = tiles(iht,1,[4*0.25,4*0.01])
print(tiles_vector)
tiles_vector = tiles(iht,1,[0,0.07])
print(tiles_vector)