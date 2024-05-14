import subprocess

gpu = 1
scene = 'bicycle'
output_dir = f'exp2_{scene}'
factor = 4

cmd = f"python train.py -s datasets/360_v2/{scene} -m {output_dir}/{scene}" \
      f" --eval -i images_{factor} --port {6109 + int(gpu)} --gpu {gpu}"

subprocess.run(cmd, shell=True)
