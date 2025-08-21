#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include <lz4.h>
#include <cstring> 
#include <unistd.h> 
#include <sys/stat.h> 
#include <chrono> 
#include <unordered_map>

#include "load_tensors.h"

using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <phase> <dtype> <model_name>" << endl;
        cerr << "  phase: 'encode' or 'decode'" << endl;
        cerr << "  dtype: 'fp16', 'bf16', 'fp16-int8', or 'bf16-int8'" << endl;
        cerr << "  model_name: e.g., 'meta-llama/Llama-3.1-8B-Instruct'" << endl;
        return 1;
    }

    string phase = argv[1];
    string dtype = argv[2];
    string model_name = argv[3];

    string global_model_dir = "/home/raunaks/benchmark_data/";
    string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/"; 
    string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/";
    string baseline_output_dir = global_model_dir + model_name + "-baseline1_lz4-" + dtype + "/";

    // --- Encoding Phase ---
    if (phase == "encode" && (dtype == "fp16" || dtype == "bf16")) {
        cout << "--- Starting LZ4 Encoding ---" << endl;
        remove_directory_if_exists(baseline_output_dir);
        create_directory_if_not_exists(baseline_output_dir);

        cout << "Loading model metadata and original tensors..." << endl;
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir);
        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>>& orig_data_vectors = model_metadata.orig_data_vectors;

        cout << "Processing " << tensor_names.size() << " tensors." << endl;
        save_tensor_names(tensor_names, baseline_output_dir); // Save names for decoding

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        cout << "Encoding 16-bit weights to single file: " << compressed_data_path << endl;

        ofstream outFile(compressed_data_path, ios::binary);
        if (!outFile) {
             cerr << "Error opening output file: " << compressed_data_path << endl;
             return 1;
        }

        auto start_encode = std::chrono::steady_clock::now();
        // uint64_t total_orig_bytes = 0;
        // uint64_t total_comp_bytes_written = 0; // Includes all headers + data
        // uint64_t total_comp_data_bytes = 0;    // Only compressed data bytes

        // --- Reuse compression buffer ---
        vector<char> compressed_buffer; // Use char* for LZ4 API

        for (const auto& tensor_name : tensor_names) {
            const vector<uint16_t>& bytes_vec = orig_data_vectors.at(tensor_name);

            uint64_t original_size_u64 = static_cast<uint64_t>(bytes_vec.size()) * sizeof(uint16_t);
            // LZ4 API uses int for sizes, check for potential overflow (unlikely for typical tensors)
            if (original_size_u64 > static_cast<uint64_t>(LZ4_MAX_INPUT_SIZE)) {
                 cerr << "Error: Tensor " << tensor_name << " size (" << original_size_u64
                      << ") exceeds LZ4_MAX_INPUT_SIZE (" << LZ4_MAX_INPUT_SIZE << ")" << endl;
                 continue;
            }
            int original_size = static_cast<int>(original_size_u64);
            // total_orig_bytes += original_size_u64;


            // --- LZ4 Compression ---
            int const max_dst_size = LZ4_compressBound(original_size);
            if (max_dst_size <= 0) {
                 cerr << "Error calculating LZ4 compress bound for tensor " << tensor_name << " size " << original_size << endl;
                 continue;
            }
            compressed_buffer.resize(max_dst_size); // Ensure buffer is large enough

            int const compressed_size = LZ4_compress_default(
                reinterpret_cast<const char*>(bytes_vec.data()), 
                compressed_buffer.data(),   
                original_size,             
                max_dst_size            
            );

            if (compressed_size <= 0) {
                cerr << "LZ4 compression failed for tensor " << tensor_name << " (return code: " << compressed_size << ")" << endl;
                continue; // Or handle error more robustly
            }

            // --- Write File Format: [orig_size][comp_size][data] ---
            uint64_t comp_size_u64 = static_cast<uint64_t>(compressed_size);

            outFile.write(reinterpret_cast<const char*>(&original_size_u64), sizeof(uint64_t)); // Write original size as uint64_t
            outFile.write(reinterpret_cast<const char*>(&comp_size_u64), sizeof(uint64_t));     // Write compressed size as uint64_t
            // Write only the actual compressed data
            outFile.write(compressed_buffer.data(), compressed_size);

            // total_comp_data_bytes += compressed_size;
            // total_comp_bytes_written += sizeof(uint64_t) + sizeof(uint64_t) + compressed_size;
        }

        outFile.flush();
        outFile.close();
        sync();

        auto end_encode = std::chrono::steady_clock::now();

        auto encoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode);
        cout << "--- Encoding Finished ---" << endl;
        cout << "Encoding time: " << encoding_elapsed.count() << " milliseconds" << endl;

        // cout << "Original Data GB: " << static_cast<double>(total_orig_bytes) / (1024 * 1024 * 1024) << endl;
        // cout << "Total Compressed Data GB: " << static_cast<double>(total_comp_data_bytes) / (1024 * 1024 * 1024) << endl;
        // cout << "Total Written (Headers + Data) GB: " << static_cast<double>(total_comp_bytes_written) / (1024 * 1024 * 1024) << endl;

        struct stat st;
        if (stat(compressed_data_path.c_str(), &st) == 0) {
            cout << "Actual Compressed File Size: " << static_cast<double>(st.st_size) / (1024 * 1024 * 1024) << " GB" << endl;
        } else {
             cerr << "Warning: Could not stat output file " << compressed_data_path << endl;
        }
    }
    else if (phase == "encode" && (dtype == "fp16-int8" || dtype == "bf16-int8")) {
        cout << "--- Starting LZ4 Encoding for " << dtype << " ---" << endl;
        remove_directory_if_exists(baseline_output_dir);
        create_directory_if_not_exists(baseline_output_dir); // Assume success

        cout << "Loading model metadata and quantized tensors..." << endl;
        // Determine source directory based on whether it's an int8 dtype or not
        string source_dtype_suffix = dtype.substr(0, dtype.find("-int8")); // fp16 or bf16
        string current_quantized_model_dir = global_model_dir + model_name + "-" + source_dtype_suffix + "-int8/";
        string current_orig_model_dir = global_model_dir + model_name + "-" + source_dtype_suffix + "/";
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(current_orig_model_dir, current_quantized_model_dir);
        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint8_t>>& quant_data_vectors = model_metadata.q_data_vectors; // Use reference

        cout << "Processing " << tensor_names.size() << " tensors." << endl;
        save_tensor_names(tensor_names, baseline_output_dir); // Save names for decoding

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        cout << "Encoding 8-bit weights to single file: " << compressed_data_path << endl;

        ofstream outFile(compressed_data_path, ios::binary);
        if (!outFile) {
             cerr << "Error opening output file: " << compressed_data_path << endl;
             return 1;
        }

        auto start_encode = std::chrono::steady_clock::now();
        vector<char> compressed_buffer; // Reuse compression buffer

        for (const auto& tensor_name : tensor_names) {
            const vector<uint8_t>& bytes_vec = quant_data_vectors.at(tensor_name); // Use .at() for safety

            uint64_t original_size_u64 = static_cast<uint64_t>(bytes_vec.size()) * sizeof(uint8_t); // Use sizeof(uint8_t)
            if (original_size_u64 > static_cast<uint64_t>(LZ4_MAX_INPUT_SIZE)) {
                 cerr << "Error: Tensor " << tensor_name << " size (" << original_size_u64
                      << ") exceeds LZ4_MAX_INPUT_SIZE (" << LZ4_MAX_INPUT_SIZE << ")" << endl;
                 continue;
            }
            int original_size = static_cast<int>(original_size_u64);

            // --- LZ4 Compression ---
            int const max_dst_size = LZ4_compressBound(original_size);
            if (max_dst_size <= 0) {
                 cerr << "Error calculating LZ4 compress bound for tensor " << tensor_name << " size " << original_size << endl;
                 continue;
            }
            compressed_buffer.resize(max_dst_size); // Ensure buffer is large enough

            int const compressed_size = LZ4_compress_default(
                reinterpret_cast<const char*>(bytes_vec.data()), // src
                compressed_buffer.data(),                       // dst
                original_size,                                  // srcSize
                max_dst_size                                    // dstCapacity
            );

            if (compressed_size <= 0) {
                cerr << "LZ4 compression failed for tensor " << tensor_name << " (return code: " << compressed_size << ")" << endl;
                continue; // Or handle error more robustly
            }

            // --- Write File Format: [orig_size][comp_size][data] ---
            uint64_t comp_size_u64 = static_cast<uint64_t>(compressed_size);

            outFile.write(reinterpret_cast<const char*>(&original_size_u64), sizeof(uint64_t));
            outFile.write(reinterpret_cast<const char*>(&comp_size_u64), sizeof(uint64_t));
            outFile.write(compressed_buffer.data(), compressed_size);
        }

        outFile.flush();
        outFile.close();
        sync();

        auto end_encode = std::chrono::steady_clock::now();
        auto encoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode);
        cout << "--- Encoding Finished ---" << endl;
        cout << "Encoding time: " << encoding_elapsed.count() << " milliseconds" << endl;

        struct stat st;
        if (stat(compressed_data_path.c_str(), &st) == 0) {
            cout << "Actual Compressed File Size: " << static_cast<double>(st.st_size) / (1024 * 1024 * 1024) << " GB" << endl;
        } else {
             cerr << "Warning: Could not stat output file " << compressed_data_path << endl;
        }
    }
    // --- Decoding Phase ---
    else if (phase == "decode" && (dtype == "fp16" || dtype == "bf16")) {
        cout << "--- Starting LZ4 Decoding (Overlapped I/O & Parallel Decomp) ---" << endl;
        string tensor_names_path = baseline_output_dir + "tensor_names.txt";
        vector<string> tensor_names;
        load_tensor_names(tensor_names_path, tensor_names); // Assume success
        cout << "Loaded " << tensor_names.size() << " tensor names." << endl;

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        ifstream compressed_file(compressed_data_path, ios::binary);
         if (!compressed_file) {
             cerr << "Error opening compressed data file: " << compressed_data_path << endl;
             return 1;
         }

        cout << "Starting overlapped reading and parallel decompression tasks..." << endl;
        auto start_overall = std::chrono::steady_clock::now();

        unordered_map<string, pair<uint16_t*, size_t>> orig_decoded_tensors;
        orig_decoded_tensors.reserve(tensor_names.size());

        int num_threads = 48; // Use available cores for tasks
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " OpenMP threads for decompression tasks." << endl;


        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp single
            {
                // cout << "I/O and Task Dispatch Thread ID: " << omp_get_thread_num() << endl;
                for (const string& tensor_name : tensor_names) {
                    uint64_t original_size = 0;
                    uint64_t compressed_size = 0;

                    // --- Sequential I/O Stage ---
                    if (!compressed_file.read(reinterpret_cast<char*>(&original_size), sizeof(uint64_t))) {
                         cerr << "Error reading original size for tensor: " << tensor_name << endl;
                         // Consider adding error handling to stop processing
                         break;
                    }
                    if (!compressed_file.read(reinterpret_cast<char*>(&compressed_size), sizeof(uint64_t))) {
                         cerr << "Error reading compressed size for tensor: " << tensor_name << endl;
                         break;
                    }

                    vector<char> task_compressed_data(compressed_size); // Use char* for LZ4
                    if (!compressed_file.read(task_compressed_data.data(), compressed_size) || compressed_file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                         cerr << "Error reading compressed data for tensor: " << tensor_name << " (expected " << compressed_size << ", got " << compressed_file.gcount() << ")" << endl;
                         break;
                    }


                    // --- Task Creation Stage ---
                    string task_tensor_name = tensor_name;
                    uint64_t task_original_size = original_size;
                    // Need compressed_size as int for LZ4 API
                    int task_compressed_size_int = static_cast<int>(compressed_size);
                     if (static_cast<uint64_t>(task_compressed_size_int) != compressed_size) {
                         cerr << "Error: Compressed size overflow for tensor " << task_tensor_name << endl;
                         break;
                     }
                     // Need original_size as int for LZ4 API
                     int task_original_size_int = static_cast<int>(task_original_size);
                     if (static_cast<uint64_t>(task_original_size_int) != task_original_size) {
                          cerr << "Error: Original size overflow for tensor " << task_tensor_name << endl;
                          break;
                     }


                    #pragma omp task default(none) \
                                    shared(orig_decoded_tensors, cerr) \
                                    firstprivate(task_tensor_name, task_compressed_data, task_original_size_int, task_compressed_size_int)
                    {
                        // --- Decompression Stage (using LZ4_decompress_safe) ---
                        if (task_original_size_int == 0) {
                             #pragma omp critical
                             {
                                 orig_decoded_tensors[task_tensor_name] = {nullptr, 0};
                             }
                        } else {
                            // Allocate buffer as char* for LZ4 API
                            char* decompressed_data_ptr_char = (char*)malloc(task_original_size_int);
                            if (!decompressed_data_ptr_char) {
                                 #pragma omp critical
                                 {
                                     cerr << "Error: malloc failed for tensor " << task_tensor_name << " size " << task_original_size_int << endl;
                                     orig_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Signal error
                                 }
                            } else {
                                int const decompressed_bytes = LZ4_decompress_safe(
                                    task_compressed_data.data(), // src
                                    decompressed_data_ptr_char,  // dst
                                    task_compressed_size_int,    // srcSize
                                    task_original_size_int       // dstCapacity
                                );

                                if (decompressed_bytes != task_original_size_int) {
                                    // Check if return value is negative (error) or positive but wrong size
                                    #pragma omp critical
                                    {
                                        if (decompressed_bytes < 0) {
                                             cerr << "LZ4 Decompression failed for tensor " << task_tensor_name << " (Error code: " << decompressed_bytes << ")" << endl;
                                        } else {
                                             cerr << "LZ4 Decompression size mismatch for tensor " << task_tensor_name << ": expected " << task_original_size_int << ", got " << decompressed_bytes << endl;
                                        }
                                    }
                                    free(decompressed_data_ptr_char);
                                    #pragma omp critical
                                    {
                                        orig_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Indicate error
                                    }
                                } 
                                else 
                                {
                                    // --- Store Result ---
                                    size_t decompressed_num_elements = task_original_size_int / sizeof(uint16_t);
                                    #pragma omp critical
                                    {
                                        orig_decoded_tensors[task_tensor_name] = {
                                            reinterpret_cast<uint16_t*>(decompressed_data_ptr_char), // Cast back to uint16_t*
                                            decompressed_num_elements
                                        };
                                    }
                                }
                            } // end else malloc succeeded
                        } // end else task_original_size > 0
                    } // End omp task
                } // End loop reading tensors

                #pragma omp taskwait
                // cout << "Single thread finished dispatching and waiting for tasks." << endl;

            } // End omp single block
        } // End omp parallel region

        compressed_file.close();

        auto end_overall = std::chrono::steady_clock::now();
        auto decoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_overall - start_overall);
        cout << "--- Decoding Finished ---" << endl;
        cout << "Overlapped I/O and Parallel Decompression finished." << endl;
        cout << "Total decoding time: " << decoding_elapsed.count() << " milliseconds" << endl;

        // --- Verification ---
        cout << "Verifying..." << endl;
        size_t successful_decodes = 0;
        bool decode_errors = false;
        for(const auto& [name, data_pair] : orig_decoded_tensors) {
            if (data_pair.second != SIZE_MAX) { // Check error signal
                successful_decodes++;
            } else {
                decode_errors = true;
            }
        }
        cout << "Successfully processed " << successful_decodes << " tensors." << endl;
        if (decode_errors || successful_decodes != tensor_names.size()) {
             cerr << "Errors occurred during decompression or not all tensors were processed." << endl;
             for (auto& [name, data_pair] : orig_decoded_tensors) {
                 if (data_pair.first) free(data_pair.first);
             }
             return 1;
        }


        cout << "Loading original tensors for verification..." << endl;
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir);
        vector<string>& actual_tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>>& orig_data_vectors = model_metadata.orig_data_vectors;

        bool verification_passed = true;
        #pragma omp parallel for num_threads(num_threads) reduction(&&:verification_passed)
        for (size_t i = 0; i < actual_tensor_names.size(); i++) {
            const auto& tensor_name = actual_tensor_names[i];

            if (orig_decoded_tensors.count(tensor_name) == 0 || orig_data_vectors.count(tensor_name) == 0) {
                 #pragma omp critical
                 {
                     cerr << "Verification Error: Tensor " << tensor_name << " missing in decoded or original data." << endl;
                 }
                 verification_passed = false;
                 continue;
            }

            const auto& decoded_pair = orig_decoded_tensors.at(tensor_name);
            const auto& orig_vec = orig_data_vectors.at(tensor_name);

            const uint16_t* decoded_ptr = decoded_pair.first;
            const size_t decoded_elements = decoded_pair.second;
            const size_t original_elements = orig_vec.size();

            bool current_tensor_ok = true;
            if (decoded_elements != original_elements) {
                 current_tensor_ok = false;
            } else if (decoded_elements > 0 && decoded_ptr != nullptr) {
                 if (memcmp(decoded_ptr, orig_vec.data(), decoded_elements * sizeof(uint16_t)) != 0) {
                     current_tensor_ok = false;
                 }
            } else if (decoded_elements == 0 && original_elements != 0) {
                 current_tensor_ok = false;
            } else if (decoded_ptr == nullptr && original_elements != 0) {
                 current_tensor_ok = false;
            }
             verification_passed = verification_passed && current_tensor_ok;
        }

        if (verification_passed) {
            cout << "Verification successful!" << endl;
        } else {
            cout << "Verification FAILED!" << endl;
        }

        // --- Cleanup ---
        cout << "Freeing decoded tensor memory..." << endl;
        size_t freed_count = 0;
        for (auto& [name, data_pair] : orig_decoded_tensors) {
            if (data_pair.first) {
                 free(data_pair.first);
                 freed_count++;
            }
        }
        cout << "Freed memory for " << freed_count << " tensors." << endl;

        if (!verification_passed) {
            return 1;
        }

    }
    else if (phase == "decode" && (dtype == "fp16-int8" || dtype == "bf16-int8")) {
        cout << "--- Starting LZ4 Decoding (Overlapped I/O & Parallel Decomp) for " << dtype << " ---" << endl;
        string tensor_names_path = baseline_output_dir + "tensor_names.txt";
        vector<string> tensor_names;
        load_tensor_names(tensor_names_path, tensor_names); // Assume success
        cout << "Loaded " << tensor_names.size() << " tensor names." << endl;

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        ifstream compressed_file(compressed_data_path, ios::binary);
         if (!compressed_file) {
             cerr << "Error opening compressed data file: " << compressed_data_path << endl;
             return 1;
         }

        cout << "Starting overlapped reading and parallel decompression tasks..." << endl;
        auto start_overall = std::chrono::steady_clock::now();

        unordered_map<string, pair<uint8_t*, size_t>> quant_decoded_tensors; // Map for uint8_t
        quant_decoded_tensors.reserve(tensor_names.size());

        int num_threads = 48; // Use available cores for tasks
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " OpenMP threads for decompression tasks." << endl;


        #pragma omp parallel
        {
            #pragma omp single
            {
                for (const string& tensor_name : tensor_names) {
                    uint64_t original_size = 0;
                    uint64_t compressed_size = 0;

                    // --- Sequential I/O Stage ---
                    if (!compressed_file.read(reinterpret_cast<char*>(&original_size), sizeof(uint64_t))) {
                         cerr << "Error reading original size for tensor: " << tensor_name << endl;
                         break;
                    }
                    if (!compressed_file.read(reinterpret_cast<char*>(&compressed_size), sizeof(uint64_t))) {
                         cerr << "Error reading compressed size for tensor: " << tensor_name << endl;
                         break;
                    }

                    if (compressed_size > static_cast<uint64_t>(numeric_limits<streamsize>::max())) {
                         cerr << "Error: Compressed size (" << compressed_size << ") too large for tensor: " << tensor_name << endl;
                         break;
                    }

                    vector<char> task_compressed_data(compressed_size);
                    if (!compressed_file.read(task_compressed_data.data(), compressed_size) || compressed_file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                         cerr << "Error reading compressed data for tensor: " << tensor_name << " (expected " << compressed_size << ", got " << compressed_file.gcount() << ")" << endl;
                         break;
                    }

                    // --- Task Creation Stage ---
                    string task_tensor_name = tensor_name;
                    uint64_t task_original_size = original_size;
                    int task_compressed_size_int = static_cast<int>(compressed_size);
                     if (static_cast<uint64_t>(task_compressed_size_int) != compressed_size) {
                         cerr << "Error: Compressed size overflow for tensor " << task_tensor_name << endl;
                         break;
                     }
                     int task_original_size_int = static_cast<int>(task_original_size);
                     if (static_cast<uint64_t>(task_original_size_int) != task_original_size) {
                          cerr << "Error: Original size overflow for tensor " << task_tensor_name << endl;
                          break;
                     }

                    #pragma omp task default(none) \
                                    shared(quant_decoded_tensors, cerr) \
                                    firstprivate(task_tensor_name, task_compressed_data, task_original_size_int, task_compressed_size_int)
                    {
                        // --- Decompression Stage (using LZ4_decompress_safe) ---
                        if (task_original_size_int == 0) {
                             #pragma omp critical
                             {
                                 quant_decoded_tensors[task_tensor_name] = {nullptr, 0};
                             }
                        } else {
                            char* decompressed_data_ptr_char = (char*)malloc(task_original_size_int);
                            if (!decompressed_data_ptr_char) {
                                 #pragma omp critical
                                 {
                                     cerr << "Error: malloc failed for tensor " << task_tensor_name << " size " << task_original_size_int << endl;
                                     quant_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Signal error
                                 }
                            } else {
                                int const decompressed_bytes = LZ4_decompress_safe(
                                    task_compressed_data.data(), // src
                                    decompressed_data_ptr_char,  // dst
                                    task_compressed_size_int,    // srcSize
                                    task_original_size_int       // dstCapacity
                                );

                                if (decompressed_bytes != task_original_size_int) {
                                    #pragma omp critical
                                    {
                                        if (decompressed_bytes < 0) {
                                             cerr << "LZ4 Decompression failed for tensor " << task_tensor_name << " (Error code: " << decompressed_bytes << ")" << endl;
                                        } else {
                                             cerr << "LZ4 Decompression size mismatch for tensor " << task_tensor_name << ": expected " << task_original_size_int << ", got " << decompressed_bytes << endl;
                                        }
                                    }
                                    free(decompressed_data_ptr_char);
                                    #pragma omp critical
                                    {
                                        quant_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Indicate error
                                    }
                                }
                                else
                                {
                                    // --- Store Result ---
                                    size_t decompressed_num_elements = task_original_size_int / sizeof(uint8_t); // Use sizeof(uint8_t)
                                    #pragma omp critical
                                    {
                                        quant_decoded_tensors[task_tensor_name] = {
                                            reinterpret_cast<uint8_t*>(decompressed_data_ptr_char), // Cast to uint8_t*
                                            decompressed_num_elements
                                        };
                                    }
                                }
                            } // end else malloc succeeded
                        } // end else task_original_size > 0
                    } // End omp task
                } // End loop reading tensors

                #pragma omp taskwait

            } // End omp single block
        } // End omp parallel region

        compressed_file.close();

        auto end_overall = std::chrono::steady_clock::now();
        auto decoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_overall - start_overall);
        cout << "--- Decoding Finished ---" << endl;
        cout << "Overlapped I/O and Parallel Decompression finished." << endl;
        cout << "Total decoding time: " << decoding_elapsed.count() << " milliseconds" << endl;

        // --- Verification ---
        cout << "Verifying..." << endl;
        size_t successful_decodes = 0;
        bool decode_errors = false;
        for(const auto& [name, data_pair] : quant_decoded_tensors) { // Iterate quant map
            if (data_pair.second != SIZE_MAX) { // Check error signal
                successful_decodes++;
            } else {
                decode_errors = true;
            }
        }
        cout << "Successfully processed " << successful_decodes << " tensors." << endl;
        if (decode_errors || successful_decodes != tensor_names.size()) {
             cerr << "Errors occurred during decompression or not all tensors were processed." << endl;
             for (auto& [name, data_pair] : quant_decoded_tensors) { // Iterate quant map
                 if (data_pair.first) free(data_pair.first);
             }
             return 1;
        }


        cout << "Loading quantized tensors for verification..." << endl;
        // Determine source directory based on whether it's an int8 dtype or not
        string source_dtype_suffix = dtype.substr(0, dtype.find("-int8")); // fp16 or bf16
        string current_quantized_model_dir = global_model_dir + model_name + "-" + source_dtype_suffix + "-int8/";
        string current_orig_model_dir = global_model_dir + model_name + "-" + source_dtype_suffix + "/";
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(current_orig_model_dir, current_quantized_model_dir);
        vector<string>& actual_tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint8_t>>& quant_data_vectors = model_metadata.q_data_vectors; // Use quant map

        bool verification_passed = true;
        #pragma omp parallel for num_threads(num_threads) reduction(&&:verification_passed)
        for (size_t i = 0; i < actual_tensor_names.size(); i++) {
            const auto& tensor_name = actual_tensor_names[i];

            if (quant_decoded_tensors.count(tensor_name) == 0 || quant_data_vectors.count(tensor_name) == 0) { // Check quant maps
                 #pragma omp critical
                 {
                     cerr << "Verification Error: Tensor " << tensor_name << " missing in decoded or quantized source data." << endl;
                 }
                 verification_passed = false;
                 continue; // Use continue inside the loop
            }

            const auto& decoded_pair = quant_decoded_tensors.at(tensor_name); // Use quant map
            const auto& quant_vec = quant_data_vectors.at(tensor_name); // Use quant map

            const uint8_t* decoded_ptr = decoded_pair.first;
            const size_t decoded_elements = decoded_pair.second;
            const size_t original_elements = quant_vec.size();

            bool current_tensor_ok = true;
            if (decoded_elements != original_elements) {
                 current_tensor_ok = false;
            } else if (decoded_elements > 0 && decoded_ptr != nullptr) {
                 if (memcmp(decoded_ptr, quant_vec.data(), decoded_elements * sizeof(uint8_t)) != 0) { // Use sizeof(uint8_t)
                     current_tensor_ok = false;
                 }
            } else if (decoded_elements == 0 && original_elements != 0) {
                 current_tensor_ok = false;
            } else if (decoded_ptr == nullptr && original_elements != 0) { // Should have been caught by SIZE_MAX check earlier
                 current_tensor_ok = false;
            }
             verification_passed = verification_passed && current_tensor_ok;
        }

        if (verification_passed) {
            cout << "Verification successful!" << endl;
        } else {
            cout << "Verification FAILED!" << endl;
        }

        // --- Cleanup ---
        cout << "Freeing decoded tensor memory..." << endl;
        size_t freed_count = 0;
        for (auto& [name, data_pair] : quant_decoded_tensors) { // Iterate quant map
            if (data_pair.first) {
                 free(data_pair.first);
                 freed_count++;
            }
        }
        cout << "Freed memory for " << freed_count << " tensors." << endl;

        if (!verification_passed) {
            return 1;
        }
    } 
    else {
        cerr << "Invalid phase ('" << phase << "') or dtype ('" << dtype << "') combination." << endl;
        return 1;
    }

    cout << "--- Program Finished Successfully ---" << endl;
    return 0;
}