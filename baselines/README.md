## Running Baselines

To compile everything, run `cmake .. && make`.

Depending on the baseline, it may be required to specify the stage (encode/decode), and data type (e.g. "bf16" for BFloat16, or "bf16-int8" for INT8 that was computed by quantizing BF16).
For all baselines, the model name needs to be specified at the top of the respective source file or bash script.
Model names used in experiments from the paper are listed in the same location and commented out.

Baseline 0 (no compression in C++):
This runs plain saving and loading in C++:

`bash baseline0_no_compression.sh`

Baseline 1 (LZ4):

`bash baseline1_lz4.sh`

Baseline 2 (ZSTD):

`bash baseline2_zstd.sh`

Baseline 3 (Safetensors):

Before running this, run `./final_encode` for qstore, which will save the index file that contains the tensor names to use.
This baseline uses the list of tensor names saved during qstore encoding to create a new safetensors file with only the relevant tensors.
Then it saves and loads this file uncompressed using safetensor's python API.

`bash baseline3_safetensors.sh`

Baseline 4 (ZipNN):

`bash baseline4_zipnn.sh`

To compute the total storage size, saving time, and loading time of the model pair, we added the respective sizes and times for 16-bit and 8-bit data.