#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include <zstd.h>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <chrono>
#include <unordered_map>

#include "load_tensors.h"

using namespace std;

// Helper function to check ZSTD errors
bool check_zstd(size_t code) {
    if (ZSTD_isError(code)) {
        cerr << "ZSTD Error: " << ZSTD_getErrorName(code) << endl;
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Needs 2 command line arguments, phase ('encode' or 'decode') and dtype ('fp16' or 'bf16')" << endl;
        return 1;
    }

    string phase = argv[1];
    string dtype = argv[2];

    // --- Configuration ---
    // string model_name = "meta-llama/Llama-3.1-8B-Instruct";
    // string model_name = "qwen/qwen2.5-7b-instruct";
    // string model_name = "mistralai/Mistral-7B-Instruct-v0.3";
    // string model_name = "qwen/qwen2.5-vl-32B-instruct";
    // string model_name = "qwen/qwen2-audio-7b-instruct";
    // string model_name = "deepseek-ai/deepseek-coder-33b-instruct";
    string model_name = "google/gemma-3-27b-it";

    string global_model_dir = "/home/raunaks/benchmark_data/";
    string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/";
    string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/";
    string baseline_output_dir = global_model_dir + model_name + "-baseline2_zstd" + "-" + dtype + "/";

    // --- Encoding Phase ---
    if (phase == "encode" && (dtype == "fp16" || dtype == "bf16")) {
        cout << "--- Starting ZSTD Encoding (Streaming API with Multi-threading) ---" << endl;
        remove_directory_if_exists(baseline_output_dir);
        create_directory_if_not_exists(baseline_output_dir);

        cout << "Loading model metadata and original tensors..." << endl;
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir);
        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>>& orig_data_vectors = model_metadata.orig_data_vectors;

        cout << "Processing " << tensor_names.size() << " tensors." << endl;
        save_tensor_names(tensor_names, baseline_output_dir);

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        cout << "Encoding 16-bit weights to single file: " << compressed_data_path << endl;

        ofstream outFile(compressed_data_path, ios::binary);
        if (!outFile) {
             cerr << "Error opening output file: " << compressed_data_path << endl;
             return 1;
        }

        // --- Create and configure ZSTD streaming compression context ---
        ZSTD_CStream* cstream = ZSTD_createCStream();
        if (!cstream) {
            cerr << "Error creating ZSTD CStream" << endl;
            return 1;
        }

        // Set parameters: Compression Level and Number of Workers
        size_t const cLevel = 2;
        int const nbWorkers = 48; // Set desired number of threads
        cout << "Using ZSTD Compression Level: " << cLevel << " with " << nbWorkers << " workers." << endl;

        if (check_zstd(ZSTD_CCtx_setParameter(cstream, ZSTD_c_compressionLevel, cLevel))) return 1;
        if (check_zstd(ZSTD_CCtx_setParameter(cstream, ZSTD_c_nbWorkers, nbWorkers))) return 1;

        auto start_encode = std::chrono::steady_clock::now();
        // uint64_t total_orig_bytes = 0;
        // uint64_t total_comp_bytes_written = 0; // Includes all headers + data
        // uint64_t total_comp_data_bytes = 0;    // Only compressed data bytes

        // --- Reuse compression buffer ---
        vector<uint8_t> compressed_buffer;
        size_t const estimated_out_buffer_size = ZSTD_CStreamOutSize(); // Recommended output buffer size

        for (const auto& tensor_name : tensor_names) {
            const vector<uint16_t>& bytes_vec = orig_data_vectors.at(tensor_name); // Use .at() for safety

            uint64_t original_size = static_cast<uint64_t>(bytes_vec.size()) * sizeof(uint16_t);
            // total_orig_bytes += original_size;

            // --- ZSTD Streaming Compression ---
            // Pledge source size for this frame (optional but can help)
            if (check_zstd(ZSTD_CCtx_setPledgedSrcSize(cstream, original_size))) return 1;


            size_t const max_dst_size = ZSTD_compressBound(original_size);
            compressed_buffer.resize(max_dst_size); // Ensure buffer is large enough

            ZSTD_inBuffer input = { bytes_vec.data(), original_size, 0 };
            ZSTD_outBuffer output = { compressed_buffer.data(), compressed_buffer.size(), 0 };

            size_t remaining = ZSTD_compressStream2(cstream, &output, &input, ZSTD_e_end);

            if (check_zstd(remaining)) {
                cerr << "Compression failed for tensor " << tensor_name << endl;
                continue; // Or handle error more robustly
            }
            if (remaining != 0) {
                // This shouldn't happen if ZSTD_compressBound is correct and output buffer is large enough
                cerr << "Error: Compression did not complete in one call for tensor " << tensor_name << ", remaining: " << remaining << endl;
                continue;
            }
            if (input.pos != input.size) {
                cerr << "Error: Not all input was consumed for tensor " << tensor_name << endl;
                continue;
            }


            size_t compressed_size = output.pos;

            // --- Write File Format: [orig_size][comp_size][data] ---
            uint64_t comp_size_u64 = static_cast<uint64_t>(compressed_size);

            outFile.write(reinterpret_cast<const char*>(&original_size), sizeof(uint64_t));
            outFile.write(reinterpret_cast<const char*>(&comp_size_u64), sizeof(uint64_t));
            // Write only the actual compressed data
            outFile.write(reinterpret_cast<const char*>(compressed_buffer.data()), compressed_size);

            // total_comp_data_bytes += compressed_size;
            // total_comp_bytes_written += sizeof(uint64_t) + sizeof(uint64_t) + compressed_size;
        }

        // --- Free the stream ---
        ZSTD_freeCStream(cstream);

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

        // Optional: Keep file size check for info
        struct stat st;
        if (stat(compressed_data_path.c_str(), &st) == 0) {
            cout << "Actual Compressed File Size: " << static_cast<double>(st.st_size) / (1024 * 1024 * 1024) << " GB" << endl;
            //  if (static_cast<uint64_t>(st.st_size) != total_comp_bytes_written) {
            //      cerr << "Warning: Calculated written bytes (" << total_comp_bytes_written
            //           << ") does not match actual file size (" << st.st_size << ")" << endl;
            //  }
        } else {
             cerr << "Warning: Could not stat output file " << compressed_data_path << endl;
        }

    }
    // --- Decoding Phase ---
    else if (phase == "decode" && (dtype == "fp16" || dtype == "bf16")) {
        cout << "--- Starting ZSTD Decoding (Overlapped I/O & Parallel Decomp) ---" << endl;
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

        int num_threads = 48; // For OpenMP tasks
        omp_set_num_threads(num_threads);

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
                         // Consider using #pragma omp cancel taskgroup or similar for robust error handling
                         break;
                    }
                    if (!compressed_file.read(reinterpret_cast<char*>(&compressed_size), sizeof(uint64_t))) {
                         cerr << "Error reading compressed size for tensor: " << tensor_name << endl;
                         break;
                    }

                    vector<uint8_t> task_compressed_data(compressed_size);
                    if (!compressed_file.read(reinterpret_cast<char*>(task_compressed_data.data()), compressed_size) || compressed_file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                         cerr << "Error reading compressed data for tensor: " << tensor_name << " (expected " << compressed_size << ", got " << compressed_file.gcount() << ")" << endl;
                         break;
                    }

                    // --- Task Creation Stage ---
                    string task_tensor_name = tensor_name;
                    uint64_t task_original_size = original_size;

                    #pragma omp task default(none) \
                                    shared(orig_decoded_tensors, cerr) \
                                    firstprivate(task_tensor_name, task_compressed_data, task_original_size)
                    {
                        // --- Decompression Stage (using single-shot ZSTD_decompress) ---
                        if (task_original_size == 0) {
                             #pragma omp critical
                             {
                                 orig_decoded_tensors[task_tensor_name] = {nullptr, 0};
                             }
                        } else {
                            uint8_t* decompressed_data_ptr = (uint8_t*)malloc(task_original_size);
                            if (!decompressed_data_ptr) {
                                 #pragma omp critical
                                 {
                                     cerr << "Error: malloc failed for tensor " << task_tensor_name << " size " << task_original_size << endl;
                                     // Signal error state if needed, e.g., by setting size to SIZE_MAX
                                     orig_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX};
                                 }
                            } else {
                                size_t decompressed_size_check = ZSTD_decompress(
                                    decompressed_data_ptr, task_original_size,
                                    task_compressed_data.data(), task_compressed_data.size()
                                );

                                if (ZSTD_isError(decompressed_size_check)) {
                                    #pragma omp critical
                                    {
                                        cerr << "Decompression failed for tensor " << task_tensor_name << ": " << ZSTD_getErrorName(decompressed_size_check) << endl;
                                    }
                                    free(decompressed_data_ptr);
                                    #pragma omp critical
                                    {
                                        orig_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Indicate error
                                    }
                                } else if (decompressed_size_check != task_original_size) {
                                    #pragma omp critical
                                    {
                                        cerr << "Decompression size mismatch for tensor " << task_tensor_name << ": expected " << task_original_size << ", got " << decompressed_size_check << endl;
                                    }
                                    free(decompressed_data_ptr);
                                    #pragma omp critical
                                    {
                                        orig_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Indicate error
                                    }
                                } else {
                                    // --- Store Result ---
                                    size_t decompressed_num_elements = task_original_size / sizeof(uint16_t);
                                    #pragma omp critical
                                    {
                                        orig_decoded_tensors[task_tensor_name] = {
                                            reinterpret_cast<uint16_t*>(decompressed_data_ptr),
                                            decompressed_num_elements
                                        };
                                    }
                                }
                            } // end else malloc succeeded
                        } // end else task_original_size > 0
                    } // End omp task
                } // End loop reading tensors

                #pragma omp taskwait
                cout << "Single thread finished dispatching and waiting for tasks." << endl;

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
        // Check for decoding errors signaled by SIZE_MAX
        size_t successful_decodes = 0;
        bool decode_errors = false;
        for(const auto& [name, data_pair] : orig_decoded_tensors) {
            if (data_pair.second != SIZE_MAX) {
                successful_decodes++;
            } else {
                decode_errors = true;
            }
        }
        cout << "Successfully processed " << successful_decodes << " tensors." << endl;
        if (decode_errors || successful_decodes != tensor_names.size()) {
             cerr << "Errors occurred during decompression or not all tensors were processed." << endl;
             // Cleanup before returning error
             for (auto& [name, data_pair] : orig_decoded_tensors) {
                 if (data_pair.first) free(data_pair.first);
             }
             return 1;
        }


        cout << "Loading original tensors for verification..." << endl;
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir);
        vector<string>& actual_tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>>& orig_data_vectors = model_metadata.orig_data_vectors;

        // Assume tensor names match from load_tensor_names
        bool verification_passed = true; // Start assuming true
        #pragma omp parallel for num_threads(num_threads) reduction(&&:verification_passed)
        for (size_t i = 0; i < actual_tensor_names.size(); i++) {
            const auto& tensor_name = actual_tensor_names[i];

            // Check existence before accessing .at()
            if (orig_decoded_tensors.count(tensor_name) == 0 || orig_data_vectors.count(tensor_name) == 0) {
                 #pragma omp critical
                 {
                     cerr << "Verification Error: Tensor " << tensor_name << " missing in decoded or original data." << endl;
                 }
                 verification_passed = false;
                 continue; // Skip comparison for this tensor (use 'continue' in parallel for)
            }


            const auto& decoded_pair = orig_decoded_tensors.at(tensor_name);
            const auto& orig_vec = orig_data_vectors.at(tensor_name);

            const uint16_t* decoded_ptr = decoded_pair.first;
            const size_t decoded_elements = decoded_pair.second;
            const size_t original_elements = orig_vec.size();

            bool current_tensor_ok = true;
            if (decoded_elements != original_elements) {
                 current_tensor_ok = false;
            } else if (decoded_elements > 0 && decoded_ptr != nullptr) { // Check ptr isn't null
                 if (memcmp(decoded_ptr, orig_vec.data(), decoded_elements * sizeof(uint16_t)) != 0) {
                     current_tensor_ok = false;
                 }
            } else if (decoded_elements == 0 && original_elements != 0) {
                 current_tensor_ok = false;
            } else if (decoded_ptr == nullptr && decoded_elements != 0) { // Check if ptr is null unexpectedly
                 current_tensor_ok = false;
            }
            // Combine result with overall verification status using reduction
             verification_passed = verification_passed && current_tensor_ok;
        } // End parallel verification loop

        if (verification_passed) {
            cout << "Verification successful!" << endl;
        } else {
            cout << "Verification FAILED!" << endl; // Still report overall failure
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
            return 1; // Return error if verification failed
        }

    }
    else if (phase == "encode" && (dtype == "fp16-int8" || dtype == "bf16-int8")) {
        cout << "--- Starting ZSTD Encoding (Streaming API with Multi-threading) for " << dtype << " ---" << endl;
        remove_directory_if_exists(baseline_output_dir);
        create_directory_if_not_exists(baseline_output_dir); // Assume success

        cout << "Loading model metadata and quantized tensors..." << endl;
        // Determine source directory based on whether it's an int8 dtype or not
        string source_dtype_suffix = dtype.substr(0, dtype.find("-int8")); // fp16 or bf16
        string current_quantized_model_dir = global_model_dir + model_name + "-" + source_dtype_suffix + "-int8/";
        string current_orig_model_dir = global_model_dir + model_name + "-" + source_dtype_suffix + "/";
        MixedPrecMetadata model_metadata = load_tensors_and_metadata(current_orig_model_dir, current_quantized_model_dir);
        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint8_t>>& quant_data_vectors = model_metadata.q_data_vectors; // Use reference for uint8_t

        cout << "Processing " << tensor_names.size() << " tensors." << endl;
        save_tensor_names(tensor_names, baseline_output_dir); // Save names for decoding

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        cout << "Encoding 8-bit weights to single file: " << compressed_data_path << endl;

        ofstream outFile(compressed_data_path, ios::binary);
        if (!outFile) {
             cerr << "Error opening output file: " << compressed_data_path << endl;
             return 1;
        }

        // --- Create and configure ZSTD streaming compression context ---
        ZSTD_CStream* cstream = ZSTD_createCStream();
        if (!cstream) {
            cerr << "Error creating ZSTD CStream" << endl;
            return 1;
        }

        // Set parameters: Compression Level and Number of Workers
        size_t const cLevel = 2;
        int const nbWorkers = 48; // Set desired number of threads
        cout << "Using ZSTD Compression Level: " << cLevel << " with " << nbWorkers << " workers." << endl;

        if (check_zstd(ZSTD_CCtx_setParameter(cstream, ZSTD_c_compressionLevel, cLevel))) return 1;
        if (check_zstd(ZSTD_CCtx_setParameter(cstream, ZSTD_c_nbWorkers, nbWorkers))) return 1;

        auto start_encode = std::chrono::steady_clock::now();

        // --- Reuse compression buffer ---
        vector<uint8_t> compressed_buffer; // Use uint8_t for consistency, though ZSTD API uses void*

        for (const auto& tensor_name : tensor_names) {
            const vector<uint8_t>& bytes_vec = quant_data_vectors.at(tensor_name); // Use uint8_t vector

            uint64_t original_size = static_cast<uint64_t>(bytes_vec.size()) * sizeof(uint8_t); // Use sizeof(uint8_t)

            // --- ZSTD Streaming Compression ---
            if (check_zstd(ZSTD_CCtx_setPledgedSrcSize(cstream, original_size))) return 1;

            size_t const max_dst_size = ZSTD_compressBound(original_size);
            compressed_buffer.resize(max_dst_size); // Ensure buffer is large enough

            ZSTD_inBuffer input = { bytes_vec.data(), original_size, 0 }; // Pass uint8_t data
            ZSTD_outBuffer output = { compressed_buffer.data(), compressed_buffer.size(), 0 };

            size_t remaining = ZSTD_compressStream2(cstream, &output, &input, ZSTD_e_end);

            if (check_zstd(remaining)) {
                cerr << "Compression failed for tensor " << tensor_name << endl;
                continue;
            }
            if (remaining != 0) {
                cerr << "Error: Compression did not complete in one call for tensor " << tensor_name << ", remaining: " << remaining << endl;
                continue;
            }
             if (input.pos != input.size) {
                 cerr << "Error: Not all input was consumed for tensor " << tensor_name << endl;
                 continue;
             }

            size_t compressed_size = output.pos;

            // --- Write File Format: [orig_size][comp_size][data] ---
            uint64_t comp_size_u64 = static_cast<uint64_t>(compressed_size);

            outFile.write(reinterpret_cast<const char*>(&original_size), sizeof(uint64_t));
            outFile.write(reinterpret_cast<const char*>(&comp_size_u64), sizeof(uint64_t));
            outFile.write(reinterpret_cast<const char*>(compressed_buffer.data()), compressed_size);
        }

        // --- Free the stream ---
        ZSTD_freeCStream(cstream);

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
    else if (phase == "decode" && (dtype == "fp16-int8" || dtype == "bf16-int8")) {
        cout << "--- Starting ZSTD Decoding (Overlapped I/O & Parallel Decomp) for " << dtype << " ---" << endl;
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

        int num_threads = 48; // For OpenMP tasks
        omp_set_num_threads(num_threads);

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

                    vector<uint8_t> task_compressed_data(compressed_size);
                    if (!compressed_file.read(reinterpret_cast<char*>(task_compressed_data.data()), compressed_size) || compressed_file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                         cerr << "Error reading compressed data for tensor: " << tensor_name << " (expected " << compressed_size << ", got " << compressed_file.gcount() << ")" << endl;
                         break;
                    }

                    // --- Task Creation Stage ---
                    string task_tensor_name = tensor_name;
                    uint64_t task_original_size = original_size;

                    #pragma omp task default(none) \
                                    shared(quant_decoded_tensors, cerr) \
                                    firstprivate(task_tensor_name, task_compressed_data, task_original_size)
                    {
                        // --- Decompression Stage (using single-shot ZSTD_decompress) ---
                        if (task_original_size == 0) {
                             #pragma omp critical
                             {
                                 quant_decoded_tensors[task_tensor_name] = {nullptr, 0};
                             }
                        } else {
                            uint8_t* decompressed_data_ptr = (uint8_t*)malloc(task_original_size); // Allocate uint8_t buffer
                            if (!decompressed_data_ptr) {
                                 #pragma omp critical
                                 {
                                     cerr << "Error: malloc failed for tensor " << task_tensor_name << " size " << task_original_size << endl;
                                     quant_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Signal error
                                 }
                            } else {
                                size_t decompressed_size_check = ZSTD_decompress(
                                    decompressed_data_ptr, task_original_size,
                                    task_compressed_data.data(), task_compressed_data.size()
                                );

                                if (ZSTD_isError(decompressed_size_check)) {
                                    #pragma omp critical
                                    {
                                        cerr << "Decompression failed for tensor " << task_tensor_name << ": " << ZSTD_getErrorName(decompressed_size_check) << endl;
                                    }
                                    free(decompressed_data_ptr);
                                    #pragma omp critical
                                    {
                                        quant_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Indicate error
                                    }
                                } else if (decompressed_size_check != task_original_size) {
                                    #pragma omp critical
                                    {
                                        cerr << "Decompression size mismatch for tensor " << task_tensor_name << ": expected " << task_original_size << ", got " << decompressed_size_check << endl;
                                    }
                                    free(decompressed_data_ptr);
                                    #pragma omp critical
                                    {
                                        quant_decoded_tensors[task_tensor_name] = {nullptr, SIZE_MAX}; // Indicate error
                                    }
                                } else {
                                    // --- Store Result ---
                                    size_t decompressed_num_elements = task_original_size / sizeof(uint8_t); // Use sizeof(uint8_t)
                                    #pragma omp critical
                                    {
                                        quant_decoded_tensors[task_tensor_name] = {
                                            decompressed_data_ptr, // Already uint8_t*
                                            decompressed_num_elements
                                        };
                                    }
                                }
                            } // end else malloc succeeded
                        } // end else task_original_size > 0
                    } // End omp task
                } // End loop reading tensors

                #pragma omp taskwait
                cout << "Single thread finished dispatching and waiting for tasks." << endl;

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
            } else if (decoded_ptr == nullptr && decoded_elements != 0) { // Should have been caught by SIZE_MAX check earlier
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
        // Keep basic phase/dtype check
        cerr << "Invalid phase ('" << phase << "') or dtype ('" << dtype << "') combination." << endl;
        return 1;
    }

    cout << "--- Program Finished Successfully ---" << endl;
    return 0;
}