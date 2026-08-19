#!/bin/bash

mkdir -p predictions

for i in {1..64}; do
    echo "[$i/64] Running DBP$i..."
    python main.py "predictions/DBP$i.txt" "DBP$i" test_seqs.txt

    if [ $? -ne 0 ]; then
        echo "ERROR: DBP$i failed"
        exit 1
    fi
done

echo "Done. All 64 predictions are in predictions/"
