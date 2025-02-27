#!/bin/bash

# First test: varying INPUT_DIM while keeping OUTPUT_DIM=1
for INPUT_DIM in 1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000
do
    # Update main.rs with new values
    sed -i "s/const INPUT_DIM: usize = [0-9]*/const INPUT_DIM: usize = $INPUT_DIM/" main.rs
    sed -i "s/const OUTPUT_DIM: usize = [0-9]*/const OUTPUT_DIM: usize = 1/" main.rs

    # Compile and run
    cargo build --release && cargo run --release
done

# Second test: varying both INPUT_DIM and OUTPUT_DIM
for INPUT_DIM in 1 10 20 30 40 50
do
    for OUTPUT_DIM in 1 10 20 30 40 50
    do
        # Update main.rs with new values
        sed -i "s/const INPUT_DIM: usize = [0-9]*/const INPUT_DIM: usize = $INPUT_DIM/" main.rs
        sed -i "s/const OUTPUT_DIM: usize = [0-9]*/const OUTPUT_DIM: usize = $OUTPUT_DIM/" main.rs

        # Compile and run
        cargo build --release && cargo run --release
    done
done