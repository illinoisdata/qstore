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

struct TensorLocationInfo {
    uint64_t offset = 0;
    uint64_t size = 0;
    bool found = false; // Flag to check if entry was found in index
};

int main(int argc, char **argv) {
    string dtype = "bf16";
    string model_name;
    
    // Check command line arguments
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <model_name>" << endl;
        cerr << "Example: " << argv[0] << " qwen/qwen2.5-vl-32B-instruct" << endl;
        return 1;
    }
    
    model_name = argv[1];
    cout << "Loading model: " << model_name << endl;

    string global_model_dir = "~/benchmark_data/";
    string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/"; // For verification
    string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/";         // For verification

    string mixedprec_output_dir = global_model_dir + model_name + "-mixedprec" + "-" + dtype + "-int8/";
    string consolidated_quant_path = mixedprec_output_dir + "all_quantized.bin";
    string consolidated_orig_path = mixedprec_output_dir + "all_original.bin";
    string index_file_path = mixedprec_output_dir + "tensor_index.tsv";

    cout << "Loading tensor index from: " << index_file_path << endl;
    unordered_map<string, pair<TensorLocationInfo, TensorLocationInfo>> tensor_locations;
    vector<string> tensor_names; // Maintain order from index file

    ifstream indexFile(index_file_path);
    if (!indexFile) {
        cerr << "Error: Could not open index file: " << index_file_path << endl;
        return 1;
    }

    string line;
    // Skip header line
    if (!getline(indexFile, line)) {
        cerr << "Error: Index file is empty or could not read header: " << index_file_path << endl;
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
            cerr << "Warning: Skipping malformed line " << line_num << " in index file (expected 4 tab-separated values): " << line << endl;
            continue;
        }

        const string& name = parts[0];
        const string& type = parts[1];
        uint64_t offset = 0;
        uint64_t size = 0;

        try {
            offset = stoull(parts[2]);
            size = stoull(parts[3]);
        } catch (const std::invalid_argument& e) {
            cerr << "Warning: Skipping line " << line_num << " due to invalid offset/size (not numbers): " << line << endl;
            continue;
        } catch (const std::out_of_range& e) {
             cerr << "Warning: Skipping line " << line_num << " due to out-of-range offset/size: " << line << endl;
             continue;
        }

        // Add tensor name to ordered list if not seen before
        if (tensor_locations.find(name) == tensor_locations.end()) {
            tensor_names.push_back(name);
            // Initialize map entry
            tensor_locations[name] = {TensorLocationInfo(), TensorLocationInfo()};
        }

        // Populate the correct part (quant or orig)
        if (type == "quant") {
            tensor_locations[name].first.offset = offset;
            tensor_locations[name].first.size = size;
            tensor_locations[name].first.found = true;
        } else if (type == "orig") {
            tensor_locations[name].second.offset = offset;
            tensor_locations[name].second.size = size;
            tensor_locations[name].second.found = true;
        } else {
            cerr << "Warning: Skipping line " << line_num << " due to unknown type '" << type << "': " << line << endl;
            // Remove name if it was just added and this was the first entry seen for it
            if (!tensor_locations[name].first.found && !tensor_locations[name].second.found) {
                 tensor_locations.erase(name);
                 if (!tensor_names.empty() && tensor_names.back() == name) {
                     tensor_names.pop_back();
                 }
            }
        }
    }
    indexFile.close();

    if (tensor_names.empty()) {
        cerr << "Error: No valid tensor entries found in index file." << endl;
        return 1;
    }
    cout << "Loaded index for " << tensor_names.size() << " tensors." << endl;


    // --- Open consolidated data files ---
    cout << "Opening consolidated data files..." << endl;
    ifstream quantInFile(consolidated_quant_path, ios::binary);
    if (!quantInFile) {
        cerr << "Error: Could not open consolidated quantized data file: " << consolidated_quant_path << endl;
        return 1;
    }
    ifstream origInFile(consolidated_orig_path, ios::binary);
    if (!origInFile) {
        cerr << "Error: Could not open consolidated original data file: " << consolidated_orig_path << endl;
        quantInFile.close(); // Close the one that did open
        return 1;
    }

    int num_threads_to_use = 48;
    omp_set_num_threads(num_threads_to_use);
    cout << "\nStarting pipelined decompression using up to " << num_threads_to_use << " threads..." << endl;
    auto start_time_decomp = std::chrono::steady_clock::now();

    // --- Pipelined Decompression ---
    unordered_map<string, vector<uint16_t>> all_decompressed_tensors_orig;
    unordered_map<string, vector<uint8_t>> all_decompressed_tensors_quant;

    vector<CompressedDataBuffer> read_buffers_orig(tensor_names.size());
    vector<CompressedDataBuffer> read_buffers_quant(tensor_names.size());

    // --- Read the first tensor's segments sequentially ---
    const string& first_tensor_name = tensor_names[0];
    cout << "Reading first tensor segments: " << first_tensor_name << endl;

    // --- Use index and read_segment_to_buffer ---
    if (tensor_locations.count(first_tensor_name)) {
        const auto& loc_pair = tensor_locations.at(first_tensor_name);
        const auto& quant_info = loc_pair.first;
        const auto& orig_info = loc_pair.second;

        if (quant_info.found) {
            read_buffers_quant[0] = read_segment_to_buffer(quantInFile, first_tensor_name, quant_info.offset, quant_info.size);
            if (!read_buffers_quant[0].success) {
                cerr << "Error reading first quantized tensor segment " << first_tensor_name << ": " << read_buffers_quant[0].error_msg << endl;
            }
        } else {
             cerr << "Warning: No 'quant' entry found in index for first tensor: " << first_tensor_name << endl;
             read_buffers_quant[0].success = false; // Mark as failed
             read_buffers_quant[0].error_msg = "Quantized entry not found in index.";
        }

        if (orig_info.found) {
            read_buffers_orig[0] = read_segment_to_buffer(origInFile, first_tensor_name, orig_info.offset, orig_info.size);
            if (!read_buffers_orig[0].success) {
                cerr << "Error reading first original tensor segment " << first_tensor_name << ": " << read_buffers_orig[0].error_msg << endl;
            }
        } else {
             cerr << "Warning: No 'orig' entry found in index for first tensor: " << first_tensor_name << endl;
             read_buffers_orig[0].success = false; // Mark as failed
             read_buffers_orig[0].error_msg = "Original entry not found in index.";
        }

    } else {
        cerr << "Error: First tensor name '" << first_tensor_name << "' not found in loaded index." << endl;
        // Mark both as failed
        read_buffers_quant[0].success = false;
        read_buffers_quant[0].error_msg = "Tensor name not found in index.";
        read_buffers_orig[0].success = false;
        read_buffers_orig[0].error_msg = "Tensor name not found in index.";
    }

    #pragma omp parallel // num_threads clause is optional if omp_set_num_threads was called
    {
        #pragma omp single nowait // One thread drives task creation and sequential reads
        {
            for (size_t i = 1; i < tensor_names.size(); ++i) {
                // --- 1. Launch Task for Previous Tensor (i-1) ---
                const string& prev_tensor_name = tensor_names[i-1];

                // Check if reads were successful
                bool can_process_orig = read_buffers_orig[i-1].success;
                bool can_process_quant = read_buffers_quant[i-1].success;

                if (can_process_orig || can_process_quant) // Launch task if at least one can be processed
                {
                    // Move buffers into the task capture
                    CompressedDataBuffer data_orig_to_process = std::move(read_buffers_orig[i-1]);
                    CompressedDataBuffer data_quant_to_process = std::move(read_buffers_quant[i-1]);

                    #pragma omp task shared(all_decompressed_tensors_orig, all_decompressed_tensors_quant) \
                    firstprivate(data_orig_to_process, data_quant_to_process, prev_tensor_name, can_process_orig, can_process_quant)
                    {
                        // --- Decompress Quantized Data ---
                        vector<uint8_t> result_quant;
                        vector<uint16_t> result_orig;
                        bool quant_decomp_success = false;
                        bool orig_decomp_success = false;
                        string task_error_msg; // To capture errors within the task

                        try {
                            if (can_process_quant) {
                                result_quant = decompress_quantized_tensor_chunked(
                                    data_quant_to_process.buffer,
                                    prev_tensor_name
                                );
                                // Check if result_quant is empty due to error or actual empty tensor
                                // The decompress function throws on error now, so success means non-error.
                                quant_decomp_success = true;
                            }

                            // --- Decompress Original Data ---
                            if (can_process_orig) {
                                if (quant_decomp_success) {
                                    result_orig = decompress_tensor_from_buffer(
                                        data_orig_to_process.buffer,
                                        prev_tensor_name,
                                        result_quant // Pass the potentially empty vector if quant failed but orig read succeeded
                                    );
                                    orig_decomp_success = true;
                                } else {
                                    // Only log if original data *could* have been processed
                                    task_error_msg = "Skipping original decompression for " + prev_tensor_name + " due to missing/failed auxiliary data (q_vec).";
                                }
                            }
                        } catch (const std::runtime_error& e) {
                             task_error_msg = "Decompression task failed for " + prev_tensor_name + ": " + e.what();
                             // Mark both as failed if exception occurs
                             quant_decomp_success = false;
                             orig_decomp_success = false;
                        } catch (const std::exception& e) {
                             task_error_msg = "Decompression task failed for " + prev_tensor_name + " with unexpected exception: " + e.what();
                             quant_decomp_success = false;
                             orig_decomp_success = false;
                        } catch (...) {
                             task_error_msg = "Decompression task failed for " + prev_tensor_name + " with unknown exception.";
                             quant_decomp_success = false;
                             orig_decomp_success = false;
                        }

                        // Store results or log errors
                        if (!task_error_msg.empty()) {
                            #pragma omp critical (cerr)
                            cerr << "Task Error: " << task_error_msg << endl;
                        }

                        if (quant_decomp_success) {
                            #pragma omp critical (decomp_results_map_quant)
                            {
                                all_decompressed_tensors_quant[prev_tensor_name] = std::move(result_quant);
                            }
                        }

                        if (orig_decomp_success) {
                            #pragma omp critical (decomp_results_map_orig)
                            {
                                all_decompressed_tensors_orig[prev_tensor_name] = std::move(result_orig);
                            }
                        }
                    } // End task for processing i-1
                } else {
                     #pragma omp critical (cerr)
                     cerr << "Skipping task creation entirely for tensor " << prev_tensor_name << " due to read errors." << endl;
                }


                // --- 2. Read Current Tensor's Segments (i) Sequentially ---
                const string& current_tensor_name = tensor_names[i];
                if (tensor_locations.count(current_tensor_name)) {
                    const auto& loc_pair = tensor_locations.at(current_tensor_name);
                    const auto& quant_info = loc_pair.first;
                    const auto& orig_info = loc_pair.second;

                    if (quant_info.found) {
                        read_buffers_quant[i] = read_segment_to_buffer(quantInFile, current_tensor_name, quant_info.offset, quant_info.size);
                        if (!read_buffers_quant[i].success) {
                            #pragma omp critical (cerr)
                            cerr << "Single Thread Error: Failed to read quantized segment for " << current_tensor_name << ": " << read_buffers_quant[i].error_msg << endl;
                        }
                    } else {
                         #pragma omp critical (cerr)
                         cerr << "Single Thread Warning: No 'quant' entry found in index for tensor: " << current_tensor_name << endl;
                         read_buffers_quant[i].success = false;
                         read_buffers_quant[i].error_msg = "Quantized entry not found in index.";
                    }

                    if (orig_info.found) {
                        read_buffers_orig[i] = read_segment_to_buffer(origInFile, current_tensor_name, orig_info.offset, orig_info.size);
                        if (!read_buffers_orig[i].success) {
                            #pragma omp critical (cerr)
                            cerr << "Single Thread Error: Failed to read original segment for " << current_tensor_name << ": " << read_buffers_orig[i].error_msg << endl;
                        }
                    } else {
                         #pragma omp critical (cerr)
                         cerr << "Single Thread Warning: No 'orig' entry found in index for tensor: " << current_tensor_name << endl;
                         read_buffers_orig[i].success = false;
                         read_buffers_orig[i].error_msg = "Original entry not found in index.";
                    }
                } else {
                     #pragma omp critical (cerr)
                     cerr << "Single Thread Error: Tensor name '" << current_tensor_name << "' not found in loaded index during sequential read." << endl;
                     read_buffers_quant[i].success = false;
                     read_buffers_quant[i].error_msg = "Tensor name not found in index.";
                     read_buffers_orig[i].success = false;
                     read_buffers_orig[i].error_msg = "Tensor name not found in index.";
                }

            } // End loop i = 1 to N-1

            // --- 3. Launch Task for the Last Tensor (N-1) ---
            size_t last_idx = tensor_names.size() - 1;
            const string& last_tensor_name = tensor_names[last_idx];

            bool can_process_orig_last = read_buffers_orig[last_idx].success;
            bool can_process_quant_last = read_buffers_quant[last_idx].success;

            if (can_process_orig_last || can_process_quant_last)
            {
                CompressedDataBuffer data_orig_to_process = std::move(read_buffers_orig[last_idx]);
                CompressedDataBuffer data_quant_to_process = std::move(read_buffers_quant[last_idx]);

                #pragma omp task shared(all_decompressed_tensors_orig, all_decompressed_tensors_quant) firstprivate(data_orig_to_process, data_quant_to_process, last_tensor_name, can_process_orig_last, can_process_quant_last)
                {
                    vector<uint8_t> result_quant;
                    vector<uint16_t> result_orig;
                    bool quant_decomp_success_last = false;
                    bool orig_decomp_success_last = false;
                    string task_error_msg_last;

                    try {
                        if (can_process_quant_last) {
                            result_quant = decompress_quantized_tensor_chunked(data_quant_to_process.buffer,
                                                                                last_tensor_name
                                                                            );
                            quant_decomp_success_last = true;
                        }

                        if (can_process_orig_last) {
                             if (quant_decomp_success_last) {
                                result_orig = decompress_tensor_from_buffer(data_orig_to_process.buffer,
                                                                            last_tensor_name,
                                                                            result_quant
                                                                        );
                                orig_decomp_success_last = true;
                            } else {
                                task_error_msg_last = "Skipping original decompression for last tensor " + last_tensor_name + " due to missing/failed auxiliary data (q_vec).";
                            }
                        }
                    } catch (const std::runtime_error& e) {
                         task_error_msg_last = "Decompression task failed for last tensor " + last_tensor_name + ": " + e.what();
                         quant_decomp_success_last = false;
                         orig_decomp_success_last = false;
                    } catch (const std::exception& e) {
                         task_error_msg_last = "Decompression task failed for last tensor " + last_tensor_name + " with unexpected exception: " + e.what();
                         quant_decomp_success_last = false;
                         orig_decomp_success_last = false;
                    } catch (...) {
                         task_error_msg_last = "Decompression task failed for last tensor " + last_tensor_name + " with unknown exception.";
                         quant_decomp_success_last = false;
                         orig_decomp_success_last = false;
                    }

                    if (!task_error_msg_last.empty()) {
                        #pragma omp critical (cerr)
                        cerr << "Task Error: " << task_error_msg_last << endl;
                    }

                    if (quant_decomp_success_last) {
                        #pragma omp critical (decomp_results_map_quant)
                        {
                            all_decompressed_tensors_quant[last_tensor_name] = std::move(result_quant);
                        }
                    }
                    if (orig_decomp_success_last) {
                        #pragma omp critical (decomp_results_map_orig)
                        {
                            all_decompressed_tensors_orig[last_tensor_name] = std::move(result_orig);
                        }
                    }
                } // End task for last tensor
            } else {
                    #pragma omp critical (cerr)
                    cerr << "Skipping task creation entirely for last tensor " << last_tensor_name << " due to read errors." << endl;
            }

            // Wait for all tasks launched by this single thread to complete
            #pragma omp taskwait

        } // End single region
    } // End parallel region

    auto end_time_decomp = std::chrono::steady_clock::now();
    auto total_elapsed_decomp = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_decomp - start_time_decomp);

    quantInFile.close();
    origInFile.close();
    cout << "Closed consolidated data files." << endl;

    cout << "\n--- Decompression Phase Results ---" << endl;
    cout << "Total pipelined decompression time: " << total_elapsed_decomp.count() << " milliseconds" << endl;
    cout << "Number of tensors attempted (from index): " << tensor_names.size() << endl;
    cout << "Number of original tensors successfully decompressed: " << all_decompressed_tensors_orig.size() << endl;
    cout << "Number of quantized tensors successfully decompressed: " << all_decompressed_tensors_quant.size() << endl;

    // --- Verification Phase (Sequential) ---
    
    cout << "\nStarting verification..." << endl;
    auto start_time_verif = std::chrono::steady_clock::now();
    size_t verified_orig_count = 0;
    size_t failed_orig_count = 0;
    size_t skipped_orig_count = 0; // Count tensors where original wasn't loaded for verification
    size_t verified_quant_count = 0;
    size_t failed_quant_count = 0;
    size_t skipped_quant_count = 0; // Count tensors where quantized wasn't loaded for verification
    size_t not_decompressed_orig_count = 0;
    size_t not_decompressed_quant_count = 0;


    // --- Load Ground Truth Data for Verification ---
    cout << "Loading ground truth models and metadata for verification..." << endl;
    MixedPrecMetadata model_metadata; // Re-use struct, load fresh data
    try {
        model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir);
    } catch (const std::exception& e) {
         cerr << "Error loading ground truth data for verification: " << e.what() << endl;
         cerr << "Skipping verification phase." << endl;
         return 1; // Cannot verify if ground truth fails to load
    }
    // No need for actual_tensor_names, we iterate through our decompressed results
    const unordered_map<string, vector<uint8_t>>& q_data_vectors_ground_truth = model_metadata.q_data_vectors;
    const unordered_map<string, vector<uint16_t>>& orig_data_vectors_ground_truth = model_metadata.orig_data_vectors;
    cout << "Ground truth data loaded." << endl;

    
    // --- Verify Original Data ---
    cout << "  Verifying original (uint16_t) tensors..." << endl;
    // Iterate through all tensors found in the index file
    for(const string& tensor_name : tensor_names) {
        // Check if it was successfully decompressed
        if (all_decompressed_tensors_orig.count(tensor_name)) {
            const vector<uint16_t>& decompressed_data = all_decompressed_tensors_orig.at(tensor_name);
            // Check if ground truth exists
            if (orig_data_vectors_ground_truth.count(tensor_name)) {
                const auto& original_data = orig_data_vectors_ground_truth.at(tensor_name);
                if (original_data.size() != decompressed_data.size()) {
                     cerr << "    Verification FAILED for " << tensor_name << " (Original): Size mismatch (Expected="
                          << original_data.size() << ", Decompressed=" << decompressed_data.size() << ")" << endl;
                     failed_orig_count++;
                } else if (memcmp(original_data.data(), decompressed_data.data(), original_data.size() * sizeof(uint16_t)) == 0) {
                    verified_orig_count++;
                } else {
                     cerr << "    Verification FAILED for " << tensor_name << " (Original): Content mismatch!" << endl;

                    failed_orig_count++;
                }
            } else {
                cout << "    Skipping verification for " << tensor_name << " (Original): Ground truth data not loaded." << endl;
                skipped_orig_count++;
            }
        } else {
            // This tensor was in the index but not successfully decompressed
            not_decompressed_orig_count++;
        }
    }

    // --- Verify Quantized Data ---
    cout << "  Verifying quantized (uint8_t) tensors..." << endl;
    for(const string& tensor_name : tensor_names) {
         // Check if it was successfully decompressed
        if (all_decompressed_tensors_quant.count(tensor_name)) {
            const vector<uint8_t>& decompressed_data = all_decompressed_tensors_quant.at(tensor_name);
            // Check if ground truth exists
            if (q_data_vectors_ground_truth.count(tensor_name)) {
                const auto& original_data = q_data_vectors_ground_truth.at(tensor_name);
                if (original_data.size() != decompressed_data.size()) {
                     cerr << "    Verification FAILED for " << tensor_name << " (Quantized): Size mismatch (Expected="
                          << original_data.size() << ", Decompressed=" << decompressed_data.size() << ")" << endl;
                     failed_quant_count++;
                } else if (memcmp(original_data.data(), decompressed_data.data(), original_data.size() * sizeof(uint8_t)) == 0) {
                    verified_quant_count++;
                } else {
                     cerr << "    Verification FAILED for " << tensor_name << " (Quantized): Content mismatch!" << endl;
                    failed_quant_count++;
                }
            } else {
                cout << "    Skipping verification for " << tensor_name << " (Quantized): Ground truth quantized data not loaded." << endl;
                skipped_quant_count++;
            }
        } else {
             // This tensor was in the index but not successfully decompressed
             not_decompressed_quant_count++;
        }
    }


    auto end_time_verif = std::chrono::steady_clock::now();
    auto total_elapsed_verif = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_verif - start_time_verif);

    cout << "\n--- Verification Phase Results ---" << endl;
    cout << "Verification time: " << total_elapsed_verif.count() << " milliseconds" << endl;
    cout << "Original Tensors:" << endl;
    cout << "  Verified successfully: " << verified_orig_count << endl;
    cout << "  Failed verification: " << failed_orig_count << endl;
    cout << "  Skipped (no ground truth): " << skipped_orig_count << endl;
    cout << "  Not decompressed (read/decomp error): " << not_decompressed_orig_count << endl;
    cout << "Quantized Tensors:" << endl;
    cout << "  Verified successfully: " << verified_quant_count << endl;
    cout << "  Failed verification: " << failed_quant_count << endl;
    cout << "  Skipped (no ground truth): " << skipped_quant_count << endl;
    cout << "  Not decompressed (read/decomp error): " << not_decompressed_quant_count << endl;


    cout << "\nDecompression and Verification finished." << endl;

    // Check if all tensors listed in the index were accounted for (verified, failed, skipped, or not decompressed)
    bool all_orig_accounted = (verified_orig_count + failed_orig_count + skipped_orig_count + not_decompressed_orig_count) == tensor_names.size();
    bool all_quant_accounted = (verified_quant_count + failed_quant_count + skipped_quant_count + not_decompressed_quant_count) == tensor_names.size();

    if (!all_orig_accounted) {
        cerr << "Error: Mismatch in original tensor verification counts!" << endl;
    }
     if (!all_quant_accounted) {
        cerr << "Error: Mismatch in quantized tensor verification counts!" << endl;
    }

    bool overall_success = (failed_orig_count == 0) &&
                           (failed_quant_count == 0) &&
                           all_orig_accounted &&
                           all_quant_accounted &&
                           (not_decompressed_orig_count == 0) && // Ideally, no tensors fail decompression
                           (not_decompressed_quant_count == 0);

    return overall_success ? 0 : 1;
}