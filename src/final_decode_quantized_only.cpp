#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <stdexcept>

extern "C" {
    #include "huf.h" // Huffman compression library (Huff0) from https://github.com/Cyan4973/FiniteStateEntropy/tree/dev/lib
}

#include "load_tensors.h"
#include "compression.h"

using namespace std;
#define MAX_THREADS_QUANTIZED_DECOMPRESSION 48

struct TensorLocationInfo {
    uint64_t offset = 0;
    uint64_t size = 0;
};

int main(int argc, char **argv) {
    string dtype = "bf16";
    // string model_name = "meta-llama/Llama-3.1-8B-Instruct";
    // string model_name = "qwen/qwen2.5-7b-instruct";
    // string model_name = "mistralai/Mistral-7B-Instruct-v0.3";
    // string model_name = "qwen/qwen2.5-vl-32B-instruct";
    // string model_name = "qwen/qwen2-audio-7b-instruct";
    // string model_name = "deepseek-ai/deepseek-coder-33b-instruct";
    string model_name = "google/gemma-3-27b-it";

    string global_model_dir = "/home/raunaks/benchmark_data/";
    string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/";
    string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/"; // Needed for verification data

    string mixedprec_output_dir = global_model_dir + model_name + "-mixedprec" + "-" + dtype + "-int8/";
    string consolidated_quant_path = mixedprec_output_dir + "all_quantized.bin";
    // string consolidated_orig_path = mixedprec_output_dir + "all_original.bin"; // Not needed for quantized-only
    string index_file_path = mixedprec_output_dir + "tensor_index.tsv";

    cout << "Loading tensor index from: " << index_file_path << endl;
    unordered_map<string, TensorLocationInfo> tensor_locations_quant; // Only store quantized info
    vector<string> tensor_names;
    ifstream indexFile(index_file_path);
    if (!indexFile) {
        cerr << "Error: Could not open index file: " << index_file_path << endl;
        return 1;
    }

    string line;
    // Skip header line
    if (!getline(indexFile, line)) {
         cerr << "Error: Could not read header line from index file: " << index_file_path << endl;
         indexFile.close();
         return 1;
    }

    int line_num = 1;
    while (getline(indexFile, line)) {
        line_num++;
        stringstream ss(line);
        string segment;
        vector<string> parts;
        while (getline(ss, segment, '\t')) {
            parts.push_back(segment);
        }

        if (parts.size() != 4) {
            cerr << "Warning: Skipping malformed line " << line_num << " in index file (expected 4 columns): " << line << endl;
            continue;
        }

        const string& name = parts[0];
        const string& type = parts[1]; // Get the type (quant or orig)

        // Only process 'quant' entries for this program
        if (type != "quant") {
            continue;
        }

        // Add tensor name to list if not seen before (for 'quant' type)
        if (tensor_locations_quant.find(name) == tensor_locations_quant.end()) {
             tensor_names.push_back(name);
        } else {
             cerr << "Warning: Duplicate 'quant' entry for tensor name '" << name << "' found in index file line " << line_num << ". Overwriting previous entry." << endl;
             // Allow overwriting if duplicate 'quant' lines exist, though ideally the index shouldn't have them.
        }


        try {
            TensorLocationInfo quant_info;
            quant_info.offset = stoull(parts[2]);
            quant_info.size = stoull(parts[3]);
            // quant_info.found = true;

            tensor_locations_quant[name] = quant_info;
            // tensor_names is populated only when a new name with type 'quant' is encountered
        } catch (const std::invalid_argument& ia) {
            cerr << "Warning: Invalid number format in index file line " << line_num << " for tensor '" << name << "'. Skipping. Error: " << ia.what() << endl;
            // Remove name if it was just added and parsing failed
            if (!tensor_names.empty() && tensor_names.back() == name) {
                 tensor_names.pop_back();
            }
            tensor_locations_quant.erase(name); // Ensure no partial entry remains
        } catch (const std::out_of_range& oor) {
            cerr << "Warning: Number out of range in index file line " << line_num << " for tensor '" << name << "'. Skipping. Error: " << oor.what() << endl;
             if (!tensor_names.empty() && tensor_names.back() == name) {
                 tensor_names.pop_back();
            }
            tensor_locations_quant.erase(name);
        }
    }
    indexFile.close();

    if (tensor_names.empty()) {
        cerr << "Error: No valid 'quant' tensor entries found in index file. Exiting." << endl;
        return -1;
    }
    cout << "Loaded index for " << tensor_names.size() << " quantized tensors." << endl;

    cout << "Opening consolidated quantized data file: " << consolidated_quant_path << endl;
    ifstream quantInFile(consolidated_quant_path, ios::binary);
    if (!quantInFile) {
        cerr << "Error: Could not open consolidated quantized file: " << consolidated_quant_path << endl;
        return 1;
    }


    // --- Configure Parallelism ---
    int num_threads_to_use = MAX_THREADS_QUANTIZED_DECOMPRESSION;
    omp_set_num_threads(num_threads_to_use);
    cout << "\nStarting quantized-only decompression using up to " << num_threads_to_use << " threads..." << endl;
    auto start_time_decomp = std::chrono::steady_clock::now();

    unordered_map<string, vector<uint8_t>> all_decompressed_tensors_quant;
    size_t read_errors = 0;
    #pragma omp parallel shared(all_decompressed_tensors_quant, tensor_names, tensor_locations_quant, consolidated_quant_path, read_errors)
    {
        #pragma omp single // One thread handles sequential reading and task launching
        {
            cout << "Reader thread (single): Opening consolidated quantized data file: " << consolidated_quant_path << endl;
            ifstream quantInFile(consolidated_quant_path, ios::binary);
            if (!quantInFile) {
                #pragma omp critical (cerr)
                cerr << "Error: Could not open consolidated quantized file in reader thread: " << consolidated_quant_path << endl;
                // Signal error - perhaps by setting a shared flag or ensuring tensor count mismatch later
                // For now, rely on task launch failures.
            } else {
                cout << "Reader thread: Launching read and decompression tasks..." << endl;
                for (size_t i = 0; i < tensor_names.size(); ++i) {
                    const string& current_tensor_name = tensor_names[i];
                    const auto& quant_loc = tensor_locations_quant.at(current_tensor_name); // Use .at() for safety

                    // Read the segment for the current tensor
                    CompressedDataBuffer data_to_process = read_segment_to_buffer(quantInFile,
                                                                                  current_tensor_name,
                                                                                  quant_loc.offset,
                                                                                  quant_loc.size);

                    if (data_to_process.success) {
                        // Move the successfully read buffer into the task capture.
                        #pragma omp task shared(all_decompressed_tensors_quant) firstprivate(data_to_process, current_tensor_name)
                        {
                            vector<uint8_t> result_quant;
                            try {
                                // Decompress the data buffer received by the task
                                result_quant = decompress_quantized_tensor_chunked(
                                    data_to_process.buffer,
                                    current_tensor_name
                                );

                                #pragma omp critical (decomp_results_map_quant)
                                {
                                    // Move the result into the map
                                    all_decompressed_tensors_quant[current_tensor_name] = std::move(result_quant);
                                }
                            } catch (const std::exception& e) {
                                 #pragma omp critical (cerr)
                                 cerr << "Task Error: Failed to decompress quantized tensor " << current_tensor_name
                                      << ": " << e.what() << endl;
                                 // Result vector remains empty, won't be added to map or size mismatch
                            }
                        } // End task for tensor i
                    } else {
                        // Reading failed for this tensor
                        #pragma omp critical (cerr)
                        cerr << "Reader thread: Error reading segment for tensor " << current_tensor_name << ": " << data_to_process.error_msg << endl;
                        #pragma omp atomic update
                        read_errors++;
                        // No task is launched for this tensor.
                    }
                } // End loop launching tasks

                quantInFile.close(); // Close file after launching all read attempts
                cout << "Reader thread: Finished reading and launching tasks. Waiting for decompression completion..." << endl;
            } // End else (file opened successfully)

            #pragma omp taskwait // Wait for all launched decompression tasks to finish
            cout << "Reader thread: All decompression tasks finished." << endl;
        } // End single region
          // Implicit barrier here: all threads wait for the single region (including taskwait) to complete.
    } // End parallel region

    auto end_time_decomp = std::chrono::steady_clock::now();
    auto total_elapsed_decomp = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_decomp - start_time_decomp);

    cout << "\n--- Quantized-Only Decompression Results ---" << endl;
    cout << "Total decompression pipeline time (read + decomp): " << total_elapsed_decomp.count() << " milliseconds" << endl;
    cout << "Number of tensors attempted: " << tensor_names.size() << endl;
    cout << "Number of tensors successfully decompressed: " << all_decompressed_tensors_quant.size() << endl;

    // --- Verification Phase (Logic Remains the Same) ---
    cout << "\nStarting verification..." << endl;
    auto start_time_verif = std::chrono::steady_clock::now();

    MixedPrecMetadata model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir); // Need both for verification data
    const unordered_map<string, vector<uint8_t>>& q_data_vectors = model_metadata.q_data_vectors;

    size_t verified_count = 0;
    size_t failed_count = 0;
    size_t skipped_count = 0; // Count tensors where original data wasn't loaded for verification

    for (const auto& pair : all_decompressed_tensors_quant) {
        const string& tensor_name = pair.first;
        const vector<uint8_t>& decompressed_data = pair.second;

        if (q_data_vectors.count(tensor_name)) {
            const auto& original_data = q_data_vectors.at(tensor_name);
            if (original_data.size() != decompressed_data.size()) {
                cerr << "  Verification FAILED for " << tensor_name << ": Size mismatch (Expected="
                        << original_data.size() << ", Got=" << decompressed_data.size() << ")" << endl;
                failed_count++;
            } else if (original_data == decompressed_data) {
                verified_count++;
            } else {
                cerr << "  Verification FAILED for " << tensor_name << ": Content mismatch" << endl;
                failed_count++;
            }
        } else {
            cout << "  Skipping verification for " << tensor_name << ": Original quantized data not loaded." << endl;
            skipped_count++;
        }
    }

    auto end_time_verif = std::chrono::steady_clock::now();
    auto total_elapsed_verif = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_verif - start_time_verif);

    cout << "Verification complete in " << total_elapsed_verif.count() << " ms" << endl;
    cout << "  Verified successfully: " << verified_count << endl;
    cout << "  Failed verification: " << failed_count << endl;
    cout << "  Skipped (no original data): " << skipped_count << endl;
    // Calculate tensors not decompressed due to read/decomp errors
    size_t not_decompressed_count = tensor_names.size() - all_decompressed_tensors_quant.size();
    cout << "  Not decompressed (read/decomp error): " << not_decompressed_count << endl;


    cout << "\nQuantized-only decompression complete." << endl;

    // Determine overall success
    bool overall_success = (failed_count == 0) && (not_decompressed_count == 0);
    cout << "Overall status: " << (overall_success ? "SUCCESS" : "FAILURE") << endl;

    return overall_success ? 0 : 1;
}