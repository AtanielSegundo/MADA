from matplotlib import pyplot as plt
from time import process_time_ns
from core.geometry import nthgone,nb_offset,offset_point,nb_offset_point,raw_offset

# Define the raw offset and nb_offset functions from the given code

# Define constants
EXTERNAL = 1
INTERNAL = -1
RAY = 10
NUMS = [10, 50, 100 , 500 , 1000]  
POINTS_LIST = [4 ,100 , 1000, 2500, 10_000, 50_000, 100_000]  

# Function to profile the raw_offset and nb_offset functions
def profile_offsets(points_list, nums):
    raw_times = {num: [] for num in nums}
    nb_times = {num: [] for num in nums}

    for points in points_list:
        geometry = nthgone(points, RAY)
        print(F"RUNNING FOR {points} POINTS")
        for num in nums:
            print(F"\tRUNNING FOR {num} NUMBER OF OFFSSETS ",end="")
            # Measure time for raw_offset
            start_time = process_time_ns()
            _ = [raw_offset(geometry, (RAY/num) * i, direction=INTERNAL) for i in range(num + 1)]
            raw_elapsed = (process_time_ns() - start_time) / 1e6
            raw_times[num].append(raw_elapsed)
            print(f"RAW: {raw_elapsed} | ",end="")
            # Measure time for nb_offset
            start_time = process_time_ns()
            _ = [nb_offset(geometry, (RAY/num) * i, direction=INTERNAL) for i in range(num + 1)]
            nb_elapsed = (process_time_ns() - start_time) / 1e6
            print(f"NB: {nb_elapsed} | ")
            nb_times[num].append(nb_elapsed)

    return raw_times, nb_times

raw_times, nb_times = profile_offsets(POINTS_LIST, NUMS)
plt.figure(figsize=(12, 8))
for num in NUMS:
    plt.plot(POINTS_LIST, raw_times[num], 'o-', label=f'PURE PYTHON OFFSETS={num}')
    plt.plot(POINTS_LIST, nb_times[num], 'x-', label=f'NUMBA OFFSETS={num}')
plt.xlabel("Number of Points")
plt.ylabel("Time Elapsed (ms)")
plt.title("Performance Comparison of raw_offset vs nb_offset")
plt.legend()
plt.grid(True)
plt.show()
