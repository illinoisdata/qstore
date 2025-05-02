#include <cassert>
#include <fstream>
#include <iostream>
// #include <set>
#include <vector>
#include <cassert>

#include <omp.h>

#include "load_tensors.h"
#include "qstore.h"
extern "C" {
#include "zipnn.h"
}

#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h> // For stat

using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Needs 2 command line arguments, phase ('encode' or 'decode') and dtype ('fp16' or 'bf16')" << endl;
        return -1;
    }

    string phase = argv[1];
    string dtype = argv[2];

    // string model_name = "meta-llama/Llama-3.1-8B-Instruct";
    // string model_name = "qwen/qwen2.5-7b-instruct";
    // string model_name = "mistralai/Mistral-7B-Instruct-v0.3";
    // string model_name = "qwen/qwen2.5-vl-32B-instruct";
    // string model_name = "qwen/qwen2-audio-7b-instruct";
    // string model_name = "deepseek-ai/deepseek-coder-33b-instruct";
    string model_name = "google/gemma-3-27b-it";

    string global_model_dir = "/home/raunaks/benchmark_data/"; // Adjust as needed
    string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/";
    string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/";
    // string mixedprec_output_dir = global_model_dir + model_name + "-mixedprec" + "-" + dtype + "-int8/";
    string baseline_output_dir = global_model_dir + model_name + "-baseline4" + "-" + dtype + "/";
    // string baseline_orig_output_dir = baseline_output_dir + "orig/"; // No longer needed

    if (phase == "encode" && (dtype == "fp16" || dtype == "bf16")) {
        try {
            // remove_directory_if_exists(baseline_orig_output_dir); // No longer needed
            create_directory_if_not_exists(baseline_output_dir); // Create base directory
        } catch (const std::exception& e) {
            cerr << "Error creating output directory: " << e.what() << endl;
            return 1;
        }

        MixedPrecMetadata model_metadata =
        load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>> orig_data_vectors = model_metadata.orig_data_vectors;

        cout << "Processing " << tensor_names.size() << " tensors." << endl;
        save_tensor_names(tensor_names, baseline_output_dir);;

        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";
        cout << "Encoding 16-bit weights to single file: " << compressed_data_path << endl;

        ofstream outFile(compressed_data_path, ios::binary);
        if (!outFile) {
            cerr << "Error opening output file: " << compressed_data_path << endl;
            return 1;
        }

        auto start = std::chrono::steady_clock::now();
        uint64_t uncomp_size = 0;
        uint64_t total_comp_bytes_written = 0; // Includes size headers
        uint64_t total_comp_data_bytes = 0; // Only compressed data

        for (size_t i = 0; i < tensor_names.size(); i++) {
            const auto& tensor_name = tensor_names[i];
            vector<uint16_t>& bytes_vec = orig_data_vectors[tensor_name];
            uncomp_size += bytes_vec.size() * 2;

            uint8_t* compressed_data = nullptr;
            size_t compressed_size = 0;
            if (!compress_uint16_data(bytes_vec.data(),
                        bytes_vec.size(),
                                &compressed_data,
                            &compressed_size,
                            48, // Number of threads for zipnn compression
                            1
                            )) {
                cerr << "Compression failed for tensor: " << tensor_name << endl;
                free(compressed_data); // Ensure cleanup even on failure before continue
                continue; // Or handle error more robustly
            }

            // Write compressed size (uint64_t)
            outFile.write(reinterpret_cast<const char*>(&compressed_size), sizeof(uint64_t));
            // Write compressed data
            outFile.write(reinterpret_cast<const char*>(compressed_data), compressed_size);

            total_comp_data_bytes += compressed_size;
            total_comp_bytes_written += sizeof(uint64_t) + compressed_size;

            free(compressed_data); // Free the buffer allocated by zipnn
        }

        outFile.flush();
        outFile.close();
        sync(); // Ensure data is written to disk

        auto end = std::chrono::steady_clock::now();
        auto encoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Encoding time: " << encoding_elapsed.count() << " milliseconds" << endl;

        cout << "Original Data GB: " << (float) uncomp_size / (1024*1024*1024) << endl;
        cout << "Total Compressed Data GB: " << (float) total_comp_data_bytes / (1024*1024*1024) << endl;
        cout << "Total Written (Data + Headers) GB: " << (float) total_comp_bytes_written / (1024*1024*1024) << endl;

        // Verify actual file size
        struct stat st;
        if (stat(compressed_data_path.c_str(), &st) == 0) {
            cout << "Actual Compressed File Size: " << (double)st.st_size / (1024 * 1024 * 1024) << " GB" << endl;
            // assert(st.st_size == total_comp_bytes_written); // Optional assertion
        } else {
            cerr << "Warning: Could not stat output data file: " << compressed_data_path << endl;
        }

    }
    else if (phase == "decode" && (dtype == "fp16" || dtype == "bf16")) {
        string tensor_names_path = baseline_output_dir + "tensor_names.txt";
        vector<string> tensor_names;
        load_tensor_names(tensor_names_path, tensor_names);
        cout << "Loaded " << tensor_names.size() << " tensor names" << endl;

        cout << "Decoding 16-bit weights (Overlapped I/O and Parallel Decompression)..." << endl; // Updated message
        auto start_overall = std::chrono::steady_clock::now();

        unordered_map<string, pair<uint16_t*, size_t>> orig_decoded_tensors;
        string compressed_data_path = baseline_output_dir + "compressed_tensors.bin";

        ifstream compressed_file(compressed_data_path, ios::binary);
        if (!compressed_file) {
            cerr << "Error opening compressed data file: " << compressed_data_path << endl;
            return 1;
        }

        cout << "Starting overlapped reading and parallel decompression tasks..." << endl;

        #pragma omp parallel num_threads(48) // Adjust thread count as needed
        {
            #pragma omp single // Use a single thread for sequential I/O and task dispatching
            {
                for (int i = 0; i < tensor_names.size(); i++) {
                    const string& tensor_name = tensor_names[i];
                    uint64_t compressed_size = 0;
                    // --- Sequential I/O Stage (for one tensor) ---
                    if (!compressed_file.read(reinterpret_cast<char*>(&compressed_size), sizeof(uint64_t))) {
                         cerr << "Error reading size header for tensor " << tensor_name << endl;
                         exit(0);
                    }

                    vector<uint8_t> task_compressed_data(compressed_size);
                    if (!compressed_file.read(reinterpret_cast<char*>(task_compressed_data.data()), compressed_size)) {
                        cerr << "Error reading " << compressed_size << " bytes for tensor " << tensor_name << endl;
                        exit(0);
                    }

                    // --- Task Creation Stage (immediately after reading) ---
                    string task_tensor_name = tensor_name; // Copy name for the task

                    #pragma omp task default(none) \
                                    shared(orig_decoded_tensors) \
                                    firstprivate(task_tensor_name, task_compressed_data) // task_compressed_data is moved implicitly by capture
                    {
                        // --- Decompression Stage (executed by a worker thread) ---
                        uint8_t* decompressed_data_ptr = nullptr;
                        size_t decompressed_size_bytes = 0;

                        // Decompress using the captured data
                        decompress_uint16_data(task_compressed_data.data(), // Use data from the captured vector
                                                 &decompressed_data_ptr,
                                                 &decompressed_size_bytes, 1, 1); // Use 1 thread for zipnn call within task

                        size_t decompressed_num_elements = decompressed_size_bytes / sizeof(uint16_t);

                        // --- Store Result (executed by worker thread) ---
                        // #pragma omp critical
                        // {
                            orig_decoded_tensors[task_tensor_name] = {
                                reinterpret_cast<uint16_t*>(decompressed_data_ptr),
                                decompressed_num_elements
                            };
                        // }
                        // task_compressed_data goes out of scope here and is destructed
                    } // End omp task
                } // End loop reading tensors and creating tasks

                // Ensure the single thread waits for all dispatched tasks before exiting the single block
                #pragma omp taskwait
            } // End omp single block
        } // End omp parallel region

        compressed_file.close();

        auto end_overall = std::chrono::steady_clock::now();
        auto decoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_overall - start_overall);
        cout << "Overlapped I/O and Parallel Decompression finished." << endl;
        cout << "Total decoding time: " << decoding_elapsed.count() << " milliseconds" << endl;

        if (orig_decoded_tensors.size() != tensor_names.size()) {
             cerr << "Error: Decoding did not complete for all tensors. Expected " << tensor_names.size() << ", got " << orig_decoded_tensors.size() << endl;
             for (auto& [name, data_pair] : orig_decoded_tensors) {
                 free(data_pair.first);
             }
             return 1;
        }
        cout << "Successfully decoded " << orig_decoded_tensors.size() << " tensors." << endl;

        cout << "Verifying..." << endl;
        MixedPrecMetadata model_metadata =
        load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

        vector<string> actual_tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>> orig_data_vectors = model_metadata.orig_data_vectors;

        // Verify tensor names match (order and content)
        assert(tensor_names.size() == actual_tensor_names.size());
        assert(tensor_names == actual_tensor_names); // Checks order too

        #pragma omp parallel for num_threads(48) // Parallel verification
        for (int i = 0; i < actual_tensor_names.size(); i++) {
            const auto& tensor_name = actual_tensor_names[i];
            const auto& decoded_pair = orig_decoded_tensors.at(tensor_name);
            const auto& orig_vec = orig_data_vectors.at(tensor_name);

            const uint16_t* data_ptr = decoded_pair.first;
            const size_t num_elements = decoded_pair.second;

            assert(num_elements == orig_vec.size());
            assert(memcmp(data_ptr, orig_vec.data(), num_elements * sizeof(uint16_t)) == 0);
            // for (size_t j = 0; j < num_elements; j++) {
            //     assert(data_ptr[j] == orig_vec[j]);
            // }
        }
        cout << "Verification successful!" << endl;

        // Free memory allocated by decompress_uint16_data
        cout << "Freeing decoded tensor memory..." << endl;
        for (auto& [name, data_pair] : orig_decoded_tensors) {
            free(data_pair.first); // Free the uint8_t* pointer allocated by zipnn
        }
        cout << "Memory freed." << endl;
    } else {
        cerr << "Invalid phase ('" << phase << "') or dtype ('" << dtype << "') combination." << endl;
        return 1;
    }
    return 0;
}