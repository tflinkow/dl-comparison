#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <epochs>"
    exit 1
fi

EPOCHS=$1

START=1.0
END=10.0
STEP=1.0

for DATA_SET in "fmnist" "cifar10" "gtsrb"; do
    if [[ "$DATA_SET" == "gtsrb" ]]; then
        LOGICS=("dl2" "g" "lk" "rc" "yg")
    else
        LOGICS=("dl2" "g" "kd" "lk" "gg" "rc" "rc-s" "rc-phi" "yg")
    fi

    wait_for_free_slots() {
        local max_slots=$1
        while true; do
            local current_jobs=$(jobs -p | wc -l)
            if (( current_jobs < max_slots )); then
                break
            fi
            sleep 1
        done
    }

    # baseline
    python3 main.py --reports-dir=reports-lambda-search --lambda-search=True --data-set=$DATA_SET --dl-weight=0 --logic=dl2 --epochs=$EPOCHS

    for logic in "${LOGICS[@]}"; do
        for weight in $(seq $START $STEP $END); do
            # choose a suitable value to run k runs in parallel
            wait_for_free_slots 6
            python3 main.py --reports-dir=reports-lambda-search --lambda-search=True --data-set=$DATA_SET --dl-weight=$weight --logic=$logic --epochs=$EPOCHS &
        done
    done

    wait

done

python lambda-search-results.py
latexmk -pdf -quiet lambda-search-plots.tex