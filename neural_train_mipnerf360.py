import subprocess
# fuck did not work?!
gpu = '1'
scene = 'bicycle'
output_dir = f'debug01_neural_{scene}'
factor = 4

voxel_size = 0.01  #0.001
update_init_factor = 16
appearance_dim = 0
ratio = 1

cmd = f"python train_neuralGS.py -s datasets/360_v2/{scene} -m {output_dir}/{scene}" \
      f" --eval -i images_{factor} --port {6109 + int(gpu)} --voxel_size {voxel_size} " \
      f"--update_init_factor {update_init_factor} --appearance_dim {appearance_dim} " \
      f"--ratio {ratio} --gpu {gpu}"

subprocess.run(cmd, shell=True)
