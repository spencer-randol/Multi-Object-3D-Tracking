""" main script for genereateing data"""
from clean_data import Clean_data_gen
from noisy_data import gen_dirty_data
from viz import viz_clean_vs_dirty_3d


SEED = 711
N = 5  # num of objects
viz = True
BOX = 10
Clean_data_gen(SEED, N, BOX)
gen_dirty_data(SEED, N, BOX)

# id turn viz off if you are generating more then
# 1-2 tru detections else to much cluster
if viz:
    viz_clean_vs_dirty_3d(SEED, N, BOX)
