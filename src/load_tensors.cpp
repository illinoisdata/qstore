#include <fstream>
#include <iostream>
#include <unordered_map>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <omp.h>
#include <chrono>

#include "nlohmann/json.hpp"
#include <nonstd/span.hpp>

#include "qstore.h"
#include "load_tensors.h"
#include "safetensors.h"

using json = nlohmann::json;
using namespace std;
namespace fs = std::filesystem;

// Helper function load_safetensor_files
void load_safetensor_files(
    const string& model_dir,
    const set<string>& unique_files,
    unordered_map<string, huggingface::safetensors::safetensors_t>& safetensors_map)
{
    vector<string> files_vec(unique_files.begin(), unique_files.end());
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < files_vec.size(); ++i) {
        const string& file_subpath = files_vec[i];
        fs::path file_path = fs::path(model_dir) / file_subpath;

        if (!fs::exists(file_path)) {
             #pragma omp critical
             cerr << "Warning: File not found, skipping: " << file_path << endl;
             continue;
        }

        ifstream file(file_path, ios::binary); // Local ifstream
        if (!file.is_open()) {
            #pragma omp critical
            cerr << "Warning: Failed to open file, skipping: " << file_path << endl;
            continue;
        }

        auto safetensors = huggingface::safetensors::deserialize(file);
        #pragma omp critical(safetensors_map_update)
        {
            safetensors_map.emplace(file_subpath, std::move(safetensors));
        }
    } // end parallel for
}


MixedPrecMetadata load_tensors_and_metadata(string orig_model_dir, string quantized_model_dir) {
    auto start_time = std::chrono::high_resolution_clock::now();
    MixedPrecMetadata metadata;

    // --- Steps 1, 2, 3: Read Index, Build Maps, Load Files ---
    cout << "Reading index files..." << endl;
    fs::path orig_index_path = fs::path(orig_model_dir) / "model.safetensors.index.json";
    fs::path q_index_path = fs::path(quantized_model_dir) / "model.safetensors.index.json";

    if (!fs::exists(orig_index_path)) throw std::runtime_error("Original index file not found: " + orig_index_path.string());
    if (!fs::exists(q_index_path)) throw std::runtime_error("Quantized index file not found: " + q_index_path.string());

    std::ifstream index_file(orig_index_path);
    std::ifstream q_index_file(q_index_path);

    json j, q_j;
    index_file >> j; 
    q_index_file >> q_j; 

    const auto &wt_map_json = j.value("weight_map", json::object());
    const auto &q_wt_map_json = q_j.value("weight_map", json::object());

    // cout << "Building file maps and identifying tensors..." << endl;
    set<string> unique_q_files, unique_orig_files, unique_scale_files;
    for (const auto &[t_name, f_name_json] : q_wt_map_json.items()) { 
        if (!f_name_json.is_string()) 
            continue;
        string f_name = f_name_json.get<string>(); 
        metadata.q_wt_file_map[t_name] = f_name; 
        unique_q_files.insert(f_name); 
        if (endsWith(t_name, ".SCB")) { 
            metadata.scale_file_map[t_name] = f_name; 
            unique_scale_files.insert(f_name); 
        } 
    }
    
    for (const auto &[t_name, f_name_json] : wt_map_json.items()) { 
        if (!f_name_json.is_string()) 
            continue; 
        string orig_f_name = f_name_json.get<string>(); 
        if (metadata.q_wt_file_map.count(t_name)) { 
            string scale_name = rstrip(t_name, ".weight") + ".SCB"; 
            if (metadata.scale_file_map.count(scale_name)) { 
                metadata.tensor_names.push_back(t_name); 
                metadata.orig_wt_file_map[t_name] = orig_f_name; 
                unique_orig_files.insert(orig_f_name); 
            } 
        }
    }

    cout << "Loading unique safetensor files..." << endl;
    load_safetensor_files(quantized_model_dir, unique_q_files, metadata.q_safetensors_map);
    load_safetensor_files(orig_model_dir, unique_orig_files, metadata.orig_safetensors_map);
    for(const auto& scale_file : unique_scale_files) { if (metadata.q_safetensors_map.count(scale_file)) { metadata.scale_safetensors_map.emplace(scale_file, metadata.q_safetensors_map.at(scale_file)); } else { cerr << "Warning: Scale file " << scale_file << " not found in loaded quantized files." << endl; } }
    auto load_files_end_time = std::chrono::high_resolution_clock::now();
    cout << "Safetensor files loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(load_files_end_time - start_time).count() << " ms." << endl;

    // --- 4. Create Intermediate Spans and Shapes ---
    unordered_map<string, span<const uint8_t>> q_data_spans_tmp;
    unordered_map<string, span<const uint16_t>> orig_data_spans_tmp;
    unordered_map<string, span<const uint32_t>> scale_data_spans_tmp;

    cout << "Extracting tensor spans and shapes..." << endl;
    #pragma omp parallel for schedule(dynamic) num_threads(48) // Adjust thread count
    for (size_t i = 0; i < metadata.tensor_names.size(); ++i) {
        const string& tensor_name = metadata.tensor_names[i];
        string scale_name = rstrip(tensor_name, ".weight") + ".SCB";

        span<const uint8_t> q_span;
        span<const uint16_t> orig_span;
        span<const uint32_t> scale_span;
        vector<size_t> shape;
        bool success = true;

        // Get Quantized Span & Shape
        const string& q_file_path = metadata.q_wt_file_map.at(tensor_name);
        const auto& q_safetensors = metadata.q_safetensors_map.at(q_file_path);
        span<const char> q_raw_span = q_safetensors[tensor_name.c_str()];
        q_span = as_typed_span<uint8_t>(q_raw_span);
        const auto& q_metadata = q_safetensors.get_metas().at(tensor_name);
        shape = q_metadata.shape;

        // Get Original Span
        const string& orig_file_path = metadata.orig_wt_file_map.at(tensor_name);
        const auto& orig_safetensors = metadata.orig_safetensors_map.at(orig_file_path);
        span<const char> orig_raw_span = orig_safetensors[tensor_name.c_str()];
        orig_span = as_typed_span<uint16_t>(orig_raw_span);

        // Get Scale Span
        const string& scale_file_path = metadata.scale_file_map.at(scale_name);
        const auto& scale_safetensors = metadata.scale_safetensors_map.at(scale_file_path);
        span<const char> scale_raw_span = scale_safetensors[scale_name.c_str()];
        scale_span = as_typed_span<uint32_t>(scale_raw_span);

        // Store intermediate spans and shape (thread-safe)
        #pragma omp critical(intermediate_data_update)
        {
            q_data_spans_tmp[tensor_name] = q_span;
            orig_data_spans_tmp[tensor_name] = orig_span;
            scale_data_spans_tmp[tensor_name] = scale_span; // May be empty if scale lookup failed
            metadata.tensor_shapes[tensor_name] = std::move(shape);
        }
    } // end parallel for


    // --- 5. Copy Data from Spans to Vectors ---
    #pragma omp parallel for schedule(dynamic) num_threads(48) // Adjust thread count
    for (size_t i = 0; i < metadata.tensor_names.size(); ++i) {
        const string& tensor_name = metadata.tensor_names[i];

        span<const uint8_t> q_span;
        span<const uint16_t> orig_span;
        span<const uint32_t> scale_span;

        #pragma omp critical(intermediate_data_update) // Read from temp maps safely
        {
            q_span = q_data_spans_tmp.at(tensor_name);
            orig_span = orig_data_spans_tmp.at(tensor_name);
            scale_span = scale_data_spans_tmp.at(tensor_name);
        }

        vector<uint8_t> q_vec(q_span.begin(), q_span.end());
        vector<uint16_t> orig_vec(orig_span.begin(), orig_span.end());
        vector<uint32_t> scale_vec(scale_span.begin(), scale_span.end());

        // Store the vectors in the final metadata maps (thread-safe)
        #pragma omp critical(vector_map_update)
        {
            metadata.q_data_vectors[tensor_name] = std::move(q_vec);
            metadata.orig_data_vectors[tensor_name] = std::move(orig_vec);
            metadata.scale_data_vectors[tensor_name] = std::move(scale_vec);
        }
    } // end parallel for

    return metadata;
}