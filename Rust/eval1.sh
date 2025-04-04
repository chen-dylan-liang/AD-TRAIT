#!/bin/bash

rm -f ./results/eval1.csv
# First test: varying INPUT_DIM while keeping OUTPUT_DIM=1
for INPUT_DIM in 1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000
do
    # Update main.rs with new values
    sed -i "s/const INPUT_DIM: usize = [0-9]*/const INPUT_DIM: usize = $INPUT_DIM/" ./src/bin/main.rs
    sed -i "s/const OUTPUT_DIM: usize = [0-9]*/const OUTPUT_DIM: usize = 1/" ./src/bin/main.rs

    # Compile and run
    cargo build --release && cargo run --release --bin main
done

# Second test: varying both INPUT_DIM and OUTPUT_DIM
for DIM in 1 10 20 30 40 50
do
        # Update main.rs with new values
        sed -i "s/const INPUT_DIM: usize = [0-9]*/const INPUT_DIM: usize = $DIM/" ./src/bin/main.rs
        sed -i "s/const OUTPUT_DIM: usize = [0-9]*/const OUTPUT_DIM: usize = $DIM/" ./src/bin/main.rs

        # Compile and run
        cargo build --release && cargo run --release --bin main
done

