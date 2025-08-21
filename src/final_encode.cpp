#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <fstream>
#include <unordered_set>
#include <filesystem> // For directory creation and size calculation
#include <unistd.h>

extern "C" {
    #include "huf.h" // Huffman compression library (Huff0) from https://github.com/Cyan4973/FiniteStateEntropy/tree/dev/lib
}

#include "load_tensors.h" // Assuming this defines MixedPrecMetadata and load_tensors_and_metadata
#include "compression.h"  // Use the updated header
#include "qstore.h"

using namespace std;
namespace fs = std::filesystem;

struct TensorLocation {
    string name;
    uint64_t offset;
    uint64_t size;
};


// --- Main Function ---
int main(int argc, char **argv) {
    // --- Configuration ---
    string dtype = "bf16";
    string model_name;
    
    // Check command line arguments
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <model_name>" << endl;
        cerr << "Example: " << argv[0] << " deepseek-ai/deepseek-coder-33b-instruct" << endl;
        return 1;
    }
    
    model_name = argv[1];
    cout << "Processing model: " << model_name << endl;

    string global_model_dir = "~/benchmark_data/";
    string quantized_model_dir = global_model_dir + model_name + "-" + dtype + "-int8/";
    string orig_model_dir = global_model_dir + model_name + "-" + dtype + "/";

    string mixedprec_output_dir = global_model_dir + model_name + "-mixedprec" + "-" + dtype + "-int8/";
    string consolidated_quant_path = mixedprec_output_dir + "all_quantized.bin";
    string consolidated_orig_path = mixedprec_output_dir + "all_original.bin";
    string index_file_path = mixedprec_output_dir + "tensor_index.tsv"; // Tab-separated index file

    try {
        if (fs::exists(consolidated_quant_path)) {
            fs::remove(consolidated_quant_path);
        }
        if (fs::exists(consolidated_orig_path)) {
            fs::remove(consolidated_orig_path);
        }
        create_directory_if_not_exists(mixedprec_output_dir);
    } catch (const std::exception& e) {
        cerr << "Error creating output directory: " << e.what() << endl;
        return 1;
    }

    MixedPrecMetadata model_metadata;
    model_metadata = load_tensors_and_metadata(orig_model_dir, quantized_model_dir);

    cout << "Metadata loaded." << endl;

    vector<string> tensor_names = model_metadata.tensor_names;
    unordered_map<string, vector<uint8_t>>& q_data_vectors = model_metadata.q_data_vectors;
    unordered_map<string, vector<uint16_t>>& orig_data_vectors = model_metadata.orig_data_vectors;
    unordered_map<string, vector<size_t>>& tensor_shapes = model_metadata.tensor_shapes;
    unordered_map<string, vector<uint32_t>>& scale_data_vectors = model_metadata.scale_data_vectors;

    size_t num_weights_in_model = 0;
    for (const auto& tensor_name: tensor_names) {
        const auto tensor_shape = tensor_shapes[tensor_name];
        num_weights_in_model += tensor_shape[0] * tensor_shape[1];
    }

    cout << "Total number of weights in model: " << num_weights_in_model << endl;

    // Optional: Process only a subset of tensors for debugging
    // tensor_names = vector<string>(tensor_names.begin(), tensor_names.begin() + 5);

    cout << "Processing " << tensor_names.size() << " tensors." << endl;

    // --- Compression Phase ---
    int max_threads = 48; // Use max available threads
    cout << "\nStarting compression using up to " << max_threads << " threads..." << endl;
    auto start_time = std::chrono::steady_clock::now();

    // --- Allocate result storage ---
    vector<vector<ChunkResult>> q_chunk_results(tensor_names.size());
    vector<vector<vector<uint8_t*>>> result_data_ptr1(tensor_names.size());
    vector<vector<vector<size_t>>> result_metadata1(tensor_names.size());
    vector<vector<vector<uint8_t*>>> result_data_ptr2(tensor_names.size());
    vector<vector<vector<size_t>>> result_metadata2(tensor_names.size());
    vector<vector<vector<uint8_t>>> result_type1(tensor_names.size());
    vector<vector<vector<uint8_t>>> result_type2(tensor_names.size());
    vector<vector<vector<size_t>>> result_original_size1(tensor_names.size());
    vector<vector<vector<size_t>>> result_original_size2(tensor_names.size());
    vector<vector<uint32_t>> row_to_group_idx_maps(tensor_names.size());
    vector<size_t> num_groups_per_tensor(tensor_names.size());

    // Pre-calculate number of groups
    for(size_t i = 0; i < tensor_names.size(); ++i) {
        const string& tensor_name = tensor_names[i];
        if (scale_data_vectors.count(tensor_name)) {
            const auto& scales = scale_data_vectors[tensor_name];
            if (scales.empty()) {
                 num_groups_per_tensor[i] = 0;
            } else {
                 unordered_set<uint32_t> unique_scales(scales.begin(), scales.end());
                 num_groups_per_tensor[i] = unique_scales.size();
            }
        } else {
            num_groups_per_tensor[i] = 0;
            cerr << "Warning: No scale data found for " << tensor_name << " during group count pre-calculation." << endl;
        }
    }

    // Allocate result storage
    for (int i = 0; i < tensor_names.size(); ++i) {
        size_t num_scale_groups = num_groups_per_tensor[i];
        if (num_scale_groups > 0) {
            result_data_ptr1[i].resize(num_scale_groups, vector<uint8_t*>(256, nullptr));
            result_metadata1[i].resize(num_scale_groups, vector<size_t>(256, 0));
            result_data_ptr2[i].resize(num_scale_groups, vector<uint8_t*>(256, nullptr));
            result_metadata2[i].resize(num_scale_groups, vector<size_t>(256, 0));
            result_type1[i].resize(num_scale_groups, vector<uint8_t>(256, TYPE_ERROR));
            result_type2[i].resize(num_scale_groups, vector<uint8_t>(256, TYPE_ERROR));
            result_original_size1[i].resize(num_scale_groups, vector<size_t>(256, 0));
            result_original_size2[i].resize(num_scale_groups, vector<size_t>(256, 0));
        }
    }
    // --- End Allocate ---

    size_t grand_total_orig_bytes = 0;
    size_t grand_total_comp_bytes = 0;
    size_t grand_total_q_orig_bytes = 0;
    size_t grand_total_q_comp_bytes = 0;

    // --- Open output files and prepare metadata storage ---
    ofstream quantOutFile(consolidated_quant_path, ios::binary | ios::trunc);
    ofstream origOutFile(consolidated_orig_path, ios::binary | ios::trunc);

    if (!quantOutFile) {
        cerr << "Error: Could not open consolidated quantized output file: " << consolidated_quant_path << endl;
        return 1;
    }
    if (!origOutFile) {
        cerr << "Error: Could not open consolidated original output file: " << consolidated_orig_path << endl;
        return 1;
    }

    vector<TensorLocation> quant_locations;
    vector<TensorLocation> orig_locations;


    // --- Main Processing Loop (Sequential over Tensors) ---
    for (int i = 0; i < tensor_names.size(); ++i) {
        const string& tensor_name = tensor_names[i];
        bool tensor_had_error = false; // Flag for errors within this tensor processing
        bool quant_serialized = false; // Track if quantized data was successfully serialized
        bool orig_serialized = false;  // Track if original data was successfully serialized

        // Check if data exists for this tensor
        if (!q_data_vectors.count(tensor_name) || !orig_data_vectors.count(tensor_name) ||
            !tensor_shapes.count(tensor_name) || !scale_data_vectors.count(tensor_name)) {
            cerr << "Warning: Skipping tensor " << tensor_name << " due to missing data (quant, orig, shape, or scale)." << endl;
            continue; // Skip to next tensor
        }

        // Get references to data AFTER checking existence
        const vector<uint8_t>& current_q_data = q_data_vectors.at(tensor_name);
        const vector<uint16_t>& current_orig_data = orig_data_vectors.at(tensor_name);
        const vector<size_t>& current_shape = tensor_shapes.at(tensor_name);
        const vector<uint32_t>& current_scales = scale_data_vectors.at(tensor_name);

        // Basic shape validation
        if (current_shape.size() != 2 || current_shape[0] == 0 || current_shape[1] == 0) {
             cerr << "Warning: Skipping tensor " << tensor_name << " due to invalid shape (not 2D or zero dimension)." << endl;
             continue;
        }
        size_t expected_elements = current_shape[0] * current_shape[1];
        if (current_q_data.size() != expected_elements || current_orig_data.size() != expected_elements || current_scales.size() != current_shape[0]) {
             cerr << "Warning: Skipping tensor " << tensor_name << " due to data size/shape mismatch." << endl;
             continue;
        }

        // cout << "\nCompressing tensor " << tensor_name << " (" << (i + 1) << "/" << tensor_names.size() << ")..." << endl;

        size_t tensor_total_orig = 0;
        size_t tensor_total_comp = 0;
        size_t tensor_q_orig_size = current_q_data.size();
        uint64_t current_q_comp_size_on_disk = 0; // Size written to file
        uint64_t current_orig_comp_size_on_disk = 0; // Size written to file

        // --- Compress Quantized Tensor ---
        q_chunk_results[i].clear();
        if (tensor_q_orig_size > 0) {
            compress_quantized_tensor_chunked(current_q_data,
                                              q_chunk_results[i],
                                              QUANTIZED_CHUNK_SIZE);

            // Check for errors in quantized compression results
            for(const auto& chunk_res : q_chunk_results[i]) {
                if (chunk_res.type == TYPE_ERROR) {
                    cerr << "  Error compressing one or more chunks of quantized tensor " << tensor_name << ". Skipping serialization for this tensor." << endl;
                    tensor_had_error = true;
                    break;
                }
            }
        } else {
             // Handle empty tensor: q_chunk_results[i] will be empty
        }

        // --- Serialize Quantized Tensor (if no compression errors) ---
        if (!tensor_had_error) {
            uint64_t offset = quantOutFile.tellp();
            uint64_t bytes_written = serialize_quantized_tensor(quantOutFile,
                                                                tensor_name,
                                                                q_chunk_results[i],
                                                                current_shape);
            if (bytes_written > 0) {
                quant_locations.push_back({tensor_name, offset, bytes_written});
                current_q_comp_size_on_disk = bytes_written;
                grand_total_q_orig_bytes += tensor_q_orig_size; // Add original size only if write succeeded
                grand_total_q_comp_bytes += current_q_comp_size_on_disk;
                quant_serialized = true;
            } else {
                cerr << "  Error serializing quantized tensor " << tensor_name << ". Skipping further processing for this tensor." << endl;
                tensor_had_error = true;
                // Note: File position might be advanced even on error, but we won't record metadata.
            }
        }


        // --- Compress Original Tensor (only if quantized part was successful/serialized) ---
        if (!tensor_had_error && num_groups_per_tensor[i] > 0) {
            compress_tensor_optimized(current_orig_data,
                            current_q_data,
                            current_scales,
                            result_data_ptr1[i],
                            result_metadata1[i],
                            result_data_ptr2[i],
                            result_metadata2[i],
                            result_type1[i],
                            result_type2[i],
                            result_original_size1[i],
                            result_original_size2[i],
                            row_to_group_idx_maps[i],
                            current_shape,
                            tensor_total_orig,
                            tensor_total_comp);

            // Check for errors reported implicitly by type
            for(size_t g=0; g<num_groups_per_tensor[i]; ++g) {
                for(int qv=0; qv<256; ++qv) {
                    if (result_type1[i][g][qv] == TYPE_ERROR || result_type2[i][g][qv] == TYPE_ERROR) {
                         cerr << "  Error during original tensor compression for " << tensor_name << " (Group: " << g << ", qVal: " << qv << "). Skipping serialization." << endl;
                         tensor_had_error = true; // Mark tensor as having an error for serialization step
                         goto end_orig_compression_check; // Use goto to break out of nested loops
                    }
                }
            }
            end_orig_compression_check:; // Label for goto

        } else if (!tensor_had_error && num_groups_per_tensor[i] == 0) {
             // Tensor is valid but has no groups (e.g., empty tensor), will serialize empty header later
             tensor_total_orig = 0;
             tensor_total_comp = 0;
        }

        // --- Serialize Original Tensor (if no compression errors) ---
        if (!tensor_had_error) {
             uint64_t offset = origOutFile.tellp();
             uint64_t bytes_written = serialize_original_tensor(origOutFile,
                                                                tensor_name,
                                                                result_data_ptr1[i],
                                                                result_metadata1[i],
                                                                result_data_ptr2[i],
                                                                result_metadata2[i],
                                                                result_type1[i],
                                                                result_type2[i],
                                                                result_original_size1[i],
                                                                result_original_size2[i],
                                                                row_to_group_idx_maps[i],
                                                                num_groups_per_tensor[i],
                                                                current_shape);
            if (bytes_written > 0) {
                orig_locations.push_back({tensor_name, offset, bytes_written});
                current_orig_comp_size_on_disk = bytes_written;
                grand_total_orig_bytes += tensor_total_orig; // Add original size only if write succeeded
                grand_total_comp_bytes += current_orig_comp_size_on_disk;
                orig_serialized = true;
            } else {
                 cerr << "  Error serializing original tensor " << tensor_name << "." << endl;
                 tensor_had_error = true;
                 // File position might be advanced, but no metadata recorded.
            }
        }


        // --- Free memory for the current tensor (AFTER serialization attempts) ---
        // Free quantized chunk results
        for (auto& chunk_res : q_chunk_results[i]) {
            if (chunk_res.data != nullptr) {
                free(chunk_res.data);
                chunk_res.data = nullptr;
            }
        }
        q_chunk_results[i].clear();
        q_chunk_results[i].shrink_to_fit();

        // Free original grouped results
        size_t groups_to_free = num_groups_per_tensor[i];
        if (groups_to_free > 0 && i < result_data_ptr1.size()) {
             if (result_data_ptr1[i].size() == groups_to_free && result_type1[i].size() == groups_to_free &&
                 result_data_ptr2[i].size() == groups_to_free && result_type2[i].size() == groups_to_free)
             {
                 for (size_t group = 0; group < groups_to_free; ++group) {
                     if (result_type1[i][group].size() == 256 && result_type2[i][group].size() == 256) {
                          for (size_t j = 0; j < 256; ++j) {
                              // Free stream 1
                              uint8_t type1 = result_type1[i][group][j];
                              uint8_t* data_ptr1 = result_data_ptr1[i][group][j];
                              if (data_ptr1 != nullptr) {
                                  if (type1 == TYPE_CHUNKED_HUF) {
                                      vector<ChunkResult>* vec_ptr = reinterpret_cast<vector<ChunkResult>*>(data_ptr1);
                                      for (auto& sub_chunk : *vec_ptr) {
                                          if (sub_chunk.data) free(sub_chunk.data);
                                          sub_chunk.data = nullptr;
                                      }
                                      delete vec_ptr;
                                  } else if (type1 == TYPE_UNCOMPRESSED || type1 == TYPE_COMPRESSED_HUF) {
                                      free(data_ptr1);
                                  }
                                  result_data_ptr1[i][group][j] = nullptr;
                              }

                              // Free stream 2
                              uint8_t type2 = result_type2[i][group][j];
                              uint8_t* data_ptr2 = result_data_ptr2[i][group][j];
                               if (data_ptr2 != nullptr) {
                                  if (type2 == TYPE_CHUNKED_HUF) {
                                      vector<ChunkResult>* vec_ptr = reinterpret_cast<vector<ChunkResult>*>(data_ptr2);
                                      for (auto& sub_chunk : *vec_ptr) {
                                          if (sub_chunk.data) free(sub_chunk.data);
                                          sub_chunk.data = nullptr;
                                      }
                                      delete vec_ptr;
                                  } else if (type2 == TYPE_UNCOMPRESSED || type2 == TYPE_COMPRESSED_HUF) {
                                      free(data_ptr2);
                                  }
                                  result_data_ptr2[i][group][j] = nullptr;
                              }
                          }
                     } else { /* Warning handled during compression/serialization */ }
                 } // end loop over groups
             } else { /* Error handled during compression/serialization */ }
        } // end if groups_to_free > 0

        row_to_group_idx_maps[i].clear();
        row_to_group_idx_maps[i].shrink_to_fit();

        if (tensor_had_error) {
             cout << "  Skipped serialization or encountered errors for tensor " << tensor_name << "." << endl;
        }

    } // End loop over tensors

    // --- Close files and write index ---
    quantOutFile.close();
    origOutFile.close();

    cout << "\nWriting tensor index file: " << index_file_path << endl;
    ofstream indexFile(index_file_path);
    if (!indexFile) {
        cerr << "Error: Failed to open index file for writing: " << index_file_path << endl;
        // Continue to show stats, but return error code later
    } else {
        // Write header
        indexFile << "tensor_name\tfile_type\toffset\tsize\n";
        // Write quantized locations
        for (const auto& loc : quant_locations) {
            indexFile << loc.name << "\tquant\t" << loc.offset << "\t" << loc.size << "\n";
        }
        // Write original locations
        for (const auto& loc : orig_locations) {
            indexFile << loc.name << "\torig\t" << loc.offset << "\t" << loc.size << "\n";
        }
        indexFile.close();
        if (!indexFile) {
             cerr << "Error: Failed to write or close index file properly." << endl;
        } else {
             cout << "Index file written successfully." << endl;
        }
    }

    sync(); // Ensure data is flushed to disk

    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    cout << "\n--- Overall Compression Results ---" << endl;
    cout << "--- Original Tensors (Grouped/Chunked) ---" << endl;
    cout << "Total Original Size (Successfully Serialized): " << (double)grand_total_orig_bytes / (1024 * 1024 * 1024) << " GB" << endl;
    cout << "Total Serialized Size (On Disk):             " << (double)grand_total_comp_bytes / (1024 * 1024 * 1024) << " GB" << endl;
    if (grand_total_orig_bytes > 0) {
        cout << "Overall Compression Ratio (vs Original):       " << static_cast<double>(grand_total_comp_bytes) / grand_total_orig_bytes << endl;
    } else {
        cout << "No original data was successfully processed or data was empty." << endl;
    }

    cout << "--- Quantized Tensors (Chunked) ---" << endl;
    cout << "Total Original Size (Successfully Serialized): " << (double)grand_total_q_orig_bytes / (1024 * 1024 * 1024) << " GB" << endl;
    cout << "Total Serialized Size (On Disk):             " << (double)grand_total_q_comp_bytes / (1024 * 1024 * 1024) << " GB" << endl;
    if (grand_total_q_orig_bytes > 0) {
        cout << "Overall Compression Ratio (vs Original):       " << static_cast<double>(grand_total_q_comp_bytes) / grand_total_q_orig_bytes << endl;
    } else {
        cout << "No quantized data was successfully processed or data was empty." << endl;
    }
    cout << "Total processing time: " << total_elapsed.count() << " milliseconds" << endl;

    cout << "\nCompression finished." << endl;

    cout << "\n--- Actual Disk Usage ---" << endl;
    try {
        size_t orig_file_size = fs::exists(consolidated_orig_path) ? fs::file_size(consolidated_orig_path) : 0;
        cout << "Original Consolidated File Size: " << (double)orig_file_size / (1024 * 1024 * 1024) << " GB" << endl;

        size_t quantized_file_size = fs::exists(consolidated_quant_path) ? fs::file_size(consolidated_quant_path) : 0;
        cout << "Quantized Consolidated File Size: " << (double)quantized_file_size / (1024 * 1024 * 1024) << " GB" << endl;

        size_t index_file_size = fs::exists(index_file_path) ? fs::file_size(index_file_path) : 0;
        cout << "Index File Size: " << (double)index_file_size / (1024 * 1024) << " MB" << endl;


        size_t total_disk_size = orig_file_size + quantized_file_size + index_file_size;
        cout << "Total MixedPrec Format Size: " << (double)total_disk_size / (1024 * 1024 * 1024) << " GB" << endl;

        // Compare disk usage to *original* uncompressed sizes loaded initially
        size_t total_uncompressed_loaded_size = 0;
         for(const auto& pair : orig_data_vectors) total_uncompressed_loaded_size += pair.second.size() * sizeof(uint16_t);
         for(const auto& pair : q_data_vectors) total_uncompressed_loaded_size += pair.second.size() * sizeof(uint8_t);

        if (total_uncompressed_loaded_size > 0) {
             cout << "Total Original Uncompressed Size (Loaded): " << (double)total_uncompressed_loaded_size / (1024 * 1024 * 1024) << " GB" << endl;
             double disk_compression_ratio = (double)total_disk_size / total_uncompressed_loaded_size;
             cout << "Overall Disk Compression Ratio (vs Loaded): " << disk_compression_ratio << endl;
        } else {
             cout << "Could not calculate overall disk compression ratio (no original data loaded)." << endl;
        }

    } catch (const std::exception& e) {
        cerr << "Error calculating disk usage: " << e.what() << endl;
    }

    // Return error if index file failed to write
    if (!indexFile && fs::exists(index_file_path)) { // Check if file stream failed after opening
        return 1;
    }

    return 0;
}