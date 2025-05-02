#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include <omp.h>

#include "load_tensors.h"
#include "qstore.h"

#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>

using namespace std;

#define NUM_THREADS_VERIFICATION 48

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Needs 2 command line arguments, phase ('encode' or 'decode') and dtype ('fp16', 'bf16', 'fp16-int8', or 'bf16-int8')" << endl;
        return -1;
    }

    string phase = argv[1];
    string dtype = argv[2];

    // string model_name = "meta-llama/Llama-3.1-8B-Instruct";
    // string model_name = "qwen/qwen2.5-7b-instruct";
    // string model_name = "mistralai/Mistral-7B-Instruct-v0.3";
    string model_name = "qwen/qwen2.5-vl-32B-instruct";
    // string model_name = "qwen/qwen2-audio-7b-instruct";
    // string model_name = "deepseek-ai/deepseek-coder-33b-instruct";
    // string model_name = "google/gemma-3-27b-it";

    string global_model_dir = "/home/raunaks/benchmark_data/";

    if (phase == "encode" && (dtype == "fp16-int8" || dtype == "bf16-int8")) {
        string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "/";
        string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/"; // Needed for loading metadata
        string baseline_output_dir = global_model_dir + model_name + "-baseline0" + "-" + dtype + "/";

        remove_directory_if_exists(baseline_output_dir);
        create_directory_if_not_exists(baseline_output_dir);

        MixedPrecMetadata model_metadata =
        load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint8_t>> q_data_vectors = model_metadata.q_data_vectors;

        cout << "Processing " << tensor_names.size() << " tensors." << endl;

        string tensor_names_path = baseline_output_dir + "tensor_names.txt";
        ofstream tensor_names_file(tensor_names_path);
        if (!tensor_names_file.is_open()) {
            cerr << "Error: Failed to open file for writing tensor names: " << tensor_names_path << endl;
            return 1;
        }
        for (const auto& name : tensor_names) {
            tensor_names_file << name << endl;
        }
        tensor_names_file.close();
        cout << "Saved " << tensor_names.size() << " tensor names to " << tensor_names_path << endl;

        string data_path = baseline_output_dir + "quantized_tensors.bin";

        cout << "Encoding 8-bit weights to single file: " << data_path << endl;
        auto start = std::chrono::steady_clock::now();

        ofstream data_file(data_path, ios::binary);
        if (!data_file.is_open()) {
            cerr << "Error: Failed to open data file for writing: " << data_path << endl;
            return 1;
        }

        // uint64_t total_written_bytes = 0; // Includes size headers
        // uint64_t total_data_bytes = 0; // Only tensor data

        // Iterate using the ordered tensor_names vector
        for (const auto& tensor_name : tensor_names) {
            vector<uint8_t>& bytes_vec = q_data_vectors[tensor_name];
            uint64_t data_size = bytes_vec.size(); // Size in bytes

            data_file.write(reinterpret_cast<const char*>(&data_size), sizeof(uint64_t));
            data_file.write(reinterpret_cast<const char*>(bytes_vec.data()), data_size);

            // total_data_bytes += data_size;
            // total_written_bytes += sizeof(uint64_t) + data_size;
        }

        data_file.flush();
        data_file.close();
        sync();

        auto end = std::chrono::steady_clock::now();
        auto encoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Encoding time: " << encoding_elapsed.count() << " milliseconds" << endl;

        // cout << "Original Data GB: " << (float) total_data_bytes / (1024*1024*1024) << endl;
        // cout << "Total written (Data + Headers) GB: " << (float) total_written_bytes / (1024*1024*1024) << endl;

        // print actual file size
        struct stat st;
        if (stat(data_path.c_str(), &st) == 0) {
            cout << "Actual Data File Size: " << (double)st.st_size / (1024 * 1024 * 1024) << " GB" << endl;
        }
    }
    else if (phase == "encode" && (dtype == "fp16" || dtype == "bf16")) {
        string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/";
        string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/"; // Needed for metadata loading
        string baseline_output_dir = global_model_dir + model_name + "-baseline0" + "-" + dtype + "/";
        
        remove_directory_if_exists(baseline_output_dir);
        create_directory_if_not_exists(baseline_output_dir);

        MixedPrecMetadata model_metadata =
        load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

        vector<string> tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>> orig_data_vectors = model_metadata.orig_data_vectors;

        cout << "Processing " << tensor_names.size() << " tensors." << endl;

        save_tensor_names(tensor_names, baseline_output_dir);

        string data_path = baseline_output_dir + "orig_tensors.bin";

        cout << "Encoding 16-bit weights to single file: " << data_path << endl;
        auto start = std::chrono::steady_clock::now();

        ofstream data_file(data_path, ios::binary);
         if (!data_file.is_open()) {
             cerr << "Error: Failed to open data file for writing: " << data_path << endl;
             return 1;
        }

        // uint64_t total_written_bytes = 0; // Includes size headers
        // uint64_t total_data_bytes = 0; // Only tensor data

        for (const auto& tensor_name : tensor_names) {
            vector<uint16_t>& bytes_vec = orig_data_vectors[tensor_name];
            uint64_t data_size = bytes_vec.size() * 2; // Size in bytes (2 bytes per element)

            data_file.write(reinterpret_cast<const char*>(&data_size), sizeof(uint64_t));
            data_file.write(reinterpret_cast<const char*>(bytes_vec.data()), data_size);

            // total_data_bytes += data_size;
            // total_written_bytes += sizeof(uint64_t) + data_size;
        }

        data_file.flush();
        data_file.close();
        sync();

        auto end = std::chrono::steady_clock::now();
        auto encoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Encoding time: " << encoding_elapsed.count() << " milliseconds" << endl;

        // cout << "Original Data GB: " << (float) total_data_bytes / (1024*1024*1024) << endl;
        // cout << "Total written (Data + Headers) GB: " << (float) total_written_bytes / (1024*1024*1024) << endl;

        struct stat st;
        if (stat(data_path.c_str(), &st) == 0) {
            cout << "Actual Data File Size: " << (double)st.st_size / (1024 * 1024 * 1024) << " GB" << endl;
        }
    }
    else if (phase == "decode" && (dtype == "fp16" || dtype == "bf16")) {
        string baseline_output_dir = global_model_dir + model_name + "-baseline0" + "-" + dtype + "/";
        string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/"; // Needed for verification
        string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/"; // Needed for verification

        string tensor_names_path = baseline_output_dir + "tensor_names.txt";
        vector<string> tensor_names;
        load_tensor_names(tensor_names_path, tensor_names);

        cout << "Loaded " << tensor_names.size() << " tensor names from " << tensor_names_path << endl;
        string data_path = baseline_output_dir + "orig_tensors.bin";

        cout << "Decoding 16-bit weights from single file: " << data_path << endl;
        auto start = std::chrono::steady_clock::now();

        ifstream data_file(data_path, ios::binary);
        if (!data_file.is_open()) {
            cerr << "Error: Failed to open data file for reading: " << data_path << endl;
            return 1;
        }

        unordered_map<string, vector<uint16_t>> decoded_data_map;

        for (const auto& tensor_name : tensor_names) {
            uint64_t data_size = 0;

            if (!data_file.read(reinterpret_cast<char *>(&data_size), sizeof(uint64_t))) {
                cerr << "Error reading size header for tensor " << tensor_name << ". Reached end of file prematurely or read error." << endl;
                return -1;
            }

            vector<uint16_t>& decoded_data = decoded_data_map[tensor_name];
            decoded_data.resize(data_size / 2);

            if (!data_file.read(reinterpret_cast<char *>(decoded_data.data()), data_size)) {
                 cerr << "Error reading " << data_size << " bytes for tensor " << tensor_name << ". Reached end of file prematurely or read error." << endl;
                 return -1;
            }
        }
        data_file.close();

        auto end = std::chrono::steady_clock::now();
        auto decoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Orig decoding time: " << decoding_elapsed.count() << " milliseconds" << endl;

        /*
        cout << endl << "Verifying..." << endl;
        MixedPrecMetadata model_metadata =
        load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

        vector<string> actual_tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint16_t>> orig_data_vectors = model_metadata.orig_data_vectors;

        if (tensor_names.size() != actual_tensor_names.size()) {
             cerr << "Verification Error: Mismatch in tensor count. Loaded names: " << tensor_names.size() << ", Original: " << actual_tensor_names.size() << endl;
             return 1;
        }

        assert(tensor_names == actual_tensor_names);

        #pragma omp parallel for num_threads(NUM_THREADS_VERIFICATION)
        for (int i = 0; i < actual_tensor_names.size(); i++) { // Iterate using original names order
            const auto& tensor_name = actual_tensor_names[i];
            const auto& decoded_data = decoded_data_map.at(tensor_name);
            const auto& orig_vec = orig_data_vectors.at(tensor_name);

            assert(decoded_data.size() == orig_vec.size());
            assert(memcmp(decoded_data.data(), orig_vec.data(), decoded_data.size() * sizeof(uint16_t)) == 0);
        }
        cout << "Verification successful!" << endl;
        */
    }
    else if (phase == "decode" && (dtype == "fp16-int8" || dtype == "bf16-int8")) {
        string baseline_output_dir = global_model_dir + model_name + "-baseline0" + "-" + dtype + "/";
        string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "/"; // Needed for verification
        string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/"; // Needed for verification loading

        string tensor_names_path = baseline_output_dir + "tensor_names.txt";
        vector<string> tensor_names;
        load_tensor_names(tensor_names_path, tensor_names);

        cout << "Loaded " << tensor_names.size() << " tensor names from " << tensor_names_path << endl;

        string data_path = baseline_output_dir + "quantized_tensors.bin";

        cout << "Decoding 8-bit weights from single file: " << data_path << endl;
        auto start = std::chrono::steady_clock::now();

        ifstream data_file(data_path, ios::binary);
        if (!data_file.is_open()) {
            cerr << "Error: Failed to open data file for reading: " << data_path << endl;
            return 1;
        }

        unordered_map<string, vector<uint8_t>> decoded_data_map;

        for (const auto& tensor_name : tensor_names) {
            uint64_t data_size = 0;

            // Read the size header
            if (!data_file.read(reinterpret_cast<char *>(&data_size), sizeof(uint64_t))) {
                 cerr << "Error reading size header for tensor " << tensor_name << ". Reached end of file prematurely or read error." << endl;
                 return -1;
            }

            // Get reference to the vector in the map
            vector<uint8_t>& decoded_data = decoded_data_map[tensor_name];
            decoded_data.resize(data_size);

            // Read the tensor data
             if (!data_file.read(reinterpret_cast<char *>(decoded_data.data()), data_size)) {
                 cerr << "Error reading " << data_size << " bytes for tensor " << tensor_name << ". Reached end of file prematurely or read error." << endl;
                 return -1;
            }
        }
        data_file.close();

        auto end = std::chrono::steady_clock::now();
        auto decoding_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Quantized decoding time: " << decoding_elapsed.count() << " milliseconds" << endl;
        /*
        cout << endl << "Verifying..." << endl;
        MixedPrecMetadata model_metadata =
        load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

        vector<string> actual_tensor_names = model_metadata.tensor_names;
        unordered_map<string, vector<uint8_t>> q_data_vectors = model_metadata.q_data_vectors;

         if (tensor_names.size() != actual_tensor_names.size()) {
             cerr << "Verification Error: Mismatch in tensor count. Loaded names: " << tensor_names.size() << ", Original: " << actual_tensor_names.size() << endl;
             return 1;
        }
        assert(tensor_names == actual_tensor_names);

        #pragma omp parallel for num_threads(NUM_THREADS_VERIFICATION)
        for (int i = 0; i < actual_tensor_names.size(); i++) { // Iterate using original names order
            const auto& tensor_name = actual_tensor_names[i];
            const auto& decoded_data = decoded_data_map.at(tensor_name);
            const auto& q_vec = q_data_vectors.at(tensor_name);

            assert(decoded_data.size() == q_vec.size());
            assert(memcmp(decoded_data.data(), q_vec.data(), decoded_data.size() * sizeof(uint8_t)) == 0);
        }
         cout << "Verification successful!" << endl;
         */
    } 
    else {
        cerr << "Invalid phase ('" << phase << "') or dtype ('" << dtype << "') combination." << endl;
        return -1;
    }

    return 0;
}