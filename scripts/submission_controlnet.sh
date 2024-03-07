
module load cuda11.3/toolkit/11.3.0
function Func1(){
    cal=1
    sleep 60
    while true
    do 
    nvidia-smi
    cal=$(($cal+1))
    if [ $cal -gt 20 ]
    then break
    fi
    sleep 2
    done
}

function Func2(){
accelerate launch --multi_gpu --mixed_precision=fp16 --main_process_port 29500 \
sd/control_net_with_sobel_small.py
}

Func1&Func2
