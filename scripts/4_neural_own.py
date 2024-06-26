# Training script for the Mip-NeRF 360 dataset

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

scenes = ["garden"]

factors = [4]

excluded_gpus = set([])

output_dir = "exp_neural_own/release"
###
# debug-01: just change voxel size=0.01
# debug-02: just debug
# debug-03: use ipdb
# debug-04: use del radii
# debug-05: remove visible_filter
# debug-06: remove visible_filter again
# debug-07: change gof2 to gof
# debug-08: add 3d filter debug (不考虑不透明度更新)
# debug-10: voxel size 0.01, offsets=5
# debug-11: radii to cpu ?work
# debug-12: del radii / to cpu()
# debug-13: radii to cpu to cuda voxel ip_filter wrong?
# debug-14: radii to cpu to cuda voxel set ipdb, not wrong?
# debug-15: global/shared
# debug-18: NAN ?
# debug-31: works but not convergence

dry_run = False
# params for scfd
voxel_size = 0.01  #0.001
update_init_factor = 16
appearance_dim = 0
ratio = 1


jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    cmd = f"CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}" \
          f" python train_neuralGS.py -s datasets/360_v2/{scene} -m {output_dir}/{scene} --eval" \
          f" -i images_{factor} --port {6109+int(gpu)}" \
          f" --voxel_size {voxel_size} --update_init_factor {update_init_factor}" \
          f" --appearance_dim {appearance_dim} --ratio {ratio}"
    # cmd = f"CUDA_VISIBLE_DEVICES={gpu}" \
    #       f" python train_neuralGS.py -s datasets/360_v2/{scene} -m {output_dir}/{scene} --eval" \
    #       f" -i images_{factor} --port {6109 + int(gpu)}" \
    #       f" --voxel_size {voxel_size} --update_init_factor {update_init_factor}" \
    #       f" --appearance_dim {appearance_dim} --ratio {ratio}"
    print(cmd)
    if not dry_run:
       os.system(cmd)

    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --data_device cpu --skip_train"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    #
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    #
    # cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh.py -m {output_dir}/{scene} --iteration 30000"
    # print(cmd)
    # if not dry_run:
    #     os.system(cmd)
    #
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1))
        # all_available_gpus = set([0,1,2,3])
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(10)
        # print('wait gpu ing')
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

