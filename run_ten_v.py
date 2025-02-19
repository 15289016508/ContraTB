import GPUtil
import time
import subprocess


def do_train_by_self(works):
    done = 0
    while done < len(works):
        GPUs = GPUtil.getGPUs()
        for gpu_item in GPUs:
            if gpu_item.id ==0:
                if gpu_item.load < 0.5 and gpu_item.memoryUtil < 0.5:
                        print('*'*20)
                        print("Training Process %d, gpu id %d" % (done + 1, gpu_item.id))
                        # st = '0,1'
                        current_params = works[done] % (str(gpu_item.id))
                        # current_params = works[done] % (st)
                        print(current_params)
                        subprocess.Popen(current_params, shell=True)
                        done += 1
                        if done >= len(works):
                            break
        time.sleep(60)
    print('done all')

# base_cmd = "nohup python train_attention.py --data_name moxifloxacin_2 --epochs 50 --batch_size 2 --lr  0.0001 --data_root data_fold/  --step_size 4000 --gpu %s --folds %s --dir_output 'output_moxifloxacin_2_jiaquan/'> nohupmoxifloxacin/%s  2>&1 &"
base_cmd = "nohup python train_lianhe.py --data_name moxifloxacin_2 --epochs 50 --batch_size 2 --lr  0.0001 --data_root data_fold/ --step_size 4000 --gpu %s --folds %s --dir_output 'output_moxifloxacin_2_lianhe/'> nohupmoxifloxacin/%s  2>&1 &"
# base_cmd = "nohup python train_sup.py --data_name moxifloxacin_2 --epochs 200  --lr  0.01 --data_root data_fold/ --batch_size 32 --step_size 800 --gpu %s --folds %s --dir_output 'output_sup/'> nohupmoxifloxacin/%s  2>&1 &"
total_jobs = []
for index in range(10):  # ten folds
    a_cur_cmd = base_cmd % ("%s", 'split_%d' % index,
                          "pre_pytorch_split_%d.txt" % (index))
    total_jobs.append(a_cur_cmd)
    print(a_cur_cmd)

do_train_by_self(total_jobs)