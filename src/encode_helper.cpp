#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdexcept>

#include <omp.h>

extern "C" {
    #include "huf.h" // Huffman compression library (Huff0) from https://github.com/Cyan4973/FiniteStateEntropy/tree/dev/lib
}

#include "compression.h" // Include the header

using namespace std;

#define MAX_THREADS_QUANTIZED_COMPRESSION 48
#define MAX_THREADS_ORIG_COMPRESSION 48

// Helper struct to manage thread-local compression buffers
struct CompressionBuffers {
    vector<uint8_t> buffer1;
    vector<uint8_t> buffer2;

    // Ensure buffer is large enough
    void ensure_capacity(size_t capacity) {
        // Add some padding to avoid frequent reallocations if sizes fluctuate slightly
        size_t desired_capacity = capacity + 1024;
        if (buffer1.size() < desired_capacity) {
            buffer1.resize(desired_capacity);
        }
         if (buffer2.size() < desired_capacity) {
            buffer2.resize(desired_capacity);
        }
    }
};

// --- Quantized Tensor Functions ---
void compress_quantized_tensor_chunked(
    const std::vector<uint8_t>& q_data_vec,
    std::vector<ChunkResult>& chunk_results,
    size_t chunk_size
) {
    size_t total_size = q_data_vec.size();
    if (total_size == 0) {
        chunk_results.clear();
        return;
    }
    size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    chunk_results.resize(num_chunks); // Pre-allocate result vector

    std::vector<std::vector<uint8_t>> thread_temp_buffers(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic) num_threads(MAX_THREADS_QUANTIZED_COMPRESSION)
    for (size_t i = 0; i < num_chunks; ++i) {
        int tid = omp_get_thread_num();
        if (tid >= thread_temp_buffers.size()) {
             #pragma omp critical (cerr)
             cerr << "Error: Thread ID " << tid << " >= buffer size " << thread_temp_buffers.size() << endl;
             chunk_results[i].type = TYPE_ERROR;
             continue;
        }

        size_t start_offset = i * chunk_size;
        size_t current_chunk_size = std::min(chunk_size, total_size - start_offset);

        // Initialize result for this chunk
        chunk_results[i].data = nullptr;
        chunk_results[i].compressed_size = 0;
        chunk_results[i].original_size = current_chunk_size;
        chunk_results[i].type = TYPE_ERROR;

        const uint8_t* chunk_src_ptr = q_data_vec.data() + start_offset;

        const size_t MIN_CHUNK_COMPRESS_SIZE = 64;
        if (current_chunk_size <= MIN_CHUNK_COMPRESS_SIZE) {
            uint8_t* final_buffer = static_cast<uint8_t*>(malloc(current_chunk_size));
            if (!final_buffer) {
                #pragma omp critical (cerr)
                cerr << "Error: malloc failed for small uncompressed quantized chunk " << i << "." << endl;
            } else {
                memcpy(final_buffer, chunk_src_ptr, current_chunk_size);
                chunk_results[i].data = final_buffer;
                chunk_results[i].compressed_size = current_chunk_size;
                chunk_results[i].type = TYPE_UNCOMPRESSED;
            }
            continue;
        }

        size_t required_tmp_size = HUF_compressBound(current_chunk_size);
        try {
            if (thread_temp_buffers[tid].size() < required_tmp_size) {
                thread_temp_buffers[tid].resize(required_tmp_size);
            }
        }
        catch (const std::bad_alloc& e) {
            #pragma omp critical (cerr)
            cerr << "Error: Failed to resize thread temp buffer for chunk " << i << ". Size: " << required_tmp_size << ". Error: " << e.what() << endl;
            continue;
        }
        uint8_t* temp_compress_buffer = thread_temp_buffers[tid].data();

        size_t c_size = HUF_compress(temp_compress_buffer, required_tmp_size,
                                     chunk_src_ptr, current_chunk_size);

        if (HUF_isError(c_size)) {
            #pragma omp critical (cerr)
            cerr << "Warning: HUF_compress failed for quantized chunk " << i << ". Error: " << HUF_getErrorName(c_size) << ". Storing uncompressed." << endl;
            c_size = 0;
        }

        if (c_size == 0 || c_size >= current_chunk_size * 0.99) {
            uint8_t* final_buffer = static_cast<uint8_t*>(malloc(current_chunk_size));
            if (!final_buffer) {
                #pragma omp critical (cerr)
                cerr << "Error: malloc failed for uncompressed quantized chunk " << i << " (after failed compression)." << endl;
            } else {
                memcpy(final_buffer, chunk_src_ptr, current_chunk_size);
                chunk_results[i].data = final_buffer;
                chunk_results[i].compressed_size = current_chunk_size;
                chunk_results[i].type = TYPE_UNCOMPRESSED;
            }
        } else {
            uint8_t* final_buffer = static_cast<uint8_t*>(malloc(c_size));
            if (!final_buffer) {
                #pragma omp critical (cerr)
                cerr << "Error: malloc failed for final compressed quantized chunk " << i << "." << endl;
            } else {
                memcpy(final_buffer, temp_compress_buffer, c_size);
                chunk_results[i].data = final_buffer;
                chunk_results[i].compressed_size = c_size;
                chunk_results[i].type = TYPE_COMPRESSED_HUF;
            }
        }
    } // End parallel for loop
}


// --- Original Tensor Compression (Modified with Chunking) ---
void compress_tensor_optimized(
    const vector<uint16_t>& input_orig_vec,
    const vector<uint8_t>& input_quantized_vec,
    const vector<uint32_t>& scale_data_vec,
    vector<vector<uint8_t*>>& result_data_ptr1,
    vector<vector<size_t>>& result_metadata1,
    vector<vector<uint8_t*>>& result_data_ptr2,
    vector<vector<size_t>>& result_metadata2,
    vector<vector<uint8_t>>& result_type1,
    vector<vector<uint8_t>>& result_type2,
    vector<vector<size_t>>& result_original_size1,
    vector<vector<size_t>>& result_original_size2,
    vector<uint32_t>& row_to_group_idx_map_out,
    const vector<size_t>& tensor_shape,
    size_t& total_orig_bytes_out,
    size_t& total_comp_bytes_out
) {
    size_t num_rows = tensor_shape[0];
    size_t num_cols = tensor_shape[1];
    size_t total_elements = num_rows * num_cols;

    if (input_orig_vec.size() != total_elements || input_quantized_vec.size() != total_elements || scale_data_vec.size() != num_rows) {
         cerr << "Error: Input vector size mismatch in compress_tensor_optimized." << endl;
         total_orig_bytes_out = 0;
         total_comp_bytes_out = 0;
         // Mark results as error? Need to ensure caller checks.
         // For now, just return zero sizes.
         return;
     }

    size_t total_orig_tensor_size_local = 0;
    size_t total_comp_tensor_size_local = 0;

    // --- Grouping Phase (Sequential) ---
    unordered_map<uint32_t, vector<size_t>> local_scale_group_indices;
    for (size_t row = 0; row < num_rows; ++row) {
        local_scale_group_indices[scale_data_vec[row]].push_back(row);
    }
    size_t num_groups = local_scale_group_indices.size();
    vector<uint32_t> unique_scales;
    unique_scales.reserve(num_groups);
    for (const auto& pair : local_scale_group_indices) { unique_scales.push_back(pair.first); }
    sort(unique_scales.begin(), unique_scales.end());
    unordered_map<uint32_t, size_t> scale_to_group_idx_map;
    for (size_t group_idx = 0; group_idx < num_groups; ++group_idx) { scale_to_group_idx_map[unique_scales[group_idx]] = group_idx; }
    row_to_group_idx_map_out.resize(num_rows);
    for (size_t row = 0; row < num_rows; ++row) {
         uint32_t scale_val = scale_data_vec[row];
         try { row_to_group_idx_map_out[row] = static_cast<uint32_t>(scale_to_group_idx_map.at(scale_val)); }
         catch (const std::out_of_range& oor) {
              cerr << "Internal Error: Scale " << scale_val << " from row " << row << " not found in map." << endl;
              // Assign a default group index (e.g., 0) or handle error appropriately
              row_to_group_idx_map_out[row] = 0; // Assuming group 0 exists or error handling is downstream
         }
    }
    // --- End Grouping Phase ---

    vector<CompressionBuffers> thread_buffers(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic) reduction(+:total_orig_tensor_size_local, total_comp_tensor_size_local) num_threads(MAX_THREADS_ORIG_COMPRESSION)
    for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
        uint32_t scale_val = unique_scales[group_idx];
        const auto& row_indices = local_scale_group_indices.at(scale_val);
        int tid = omp_get_thread_num();
         if (tid >= thread_buffers.size()) {
             #pragma omp critical (cerr)
             cerr << "Error: Thread ID " << tid << " >= buffer size " << thread_buffers.size() << " in original compression." << endl;
             // Mark results for this group as error?
             continue; // Skip this group
         }
        if (row_indices.empty()) continue;

        // --- Permutation Phase ---
        vector<vector<uint8_t>> permuted_vec1(256);
        vector<vector<uint8_t>> permuted_vec2(256);
        vector<size_t> counts(256, 0);
        size_t group_total_elements = 0;
        for (size_t row_idx : row_indices) {
            size_t start_index = row_idx * num_cols;
            group_total_elements += num_cols;
            size_t end_index = start_index + num_cols;
            for (size_t idx = start_index; idx < end_index; ++idx) { counts[input_quantized_vec[idx]]++; }
        }
        if (group_total_elements == 0) continue;
        for (int q_val = 0; q_val < 256; ++q_val) {
            if (counts[q_val] > 0) {
                try {
                    permuted_vec1[q_val].reserve(counts[q_val]);
                    permuted_vec2[q_val].reserve(counts[q_val]);
                } catch (const std::bad_alloc& e) {
                    #pragma omp critical (cerr)
                    cerr << "Error: Failed to reserve memory for permuted vectors. GroupIdx: " << group_idx << ", qVal: " << q_val << ". Error: " << e.what() << endl;
                    counts[q_val] = 0; // Mark count as 0 so we don't try to access later
                }
            }
        }
        for (size_t row_idx : row_indices) {
            size_t start_index = row_idx * num_cols;
            size_t end_index = start_index + num_cols;
            for (size_t idx = start_index; idx < end_index; ++idx) {
                uint8_t q_wt = input_quantized_vec[idx];
                if (counts[q_wt] == 0) continue; // Skip if reservation failed
                uint16_t orig_val = input_orig_vec[idx];
                permuted_vec1[q_wt].push_back(static_cast<uint8_t>((orig_val >> 8) & 0xFF));
                permuted_vec2[q_wt].push_back(static_cast<uint8_t>(orig_val & 0xFF));
            }
        }
        // --- End Permutation Phase ---

        // --- Compression Phase (With Chunking Logic) ---
        size_t group_orig_bytes_local = 0;
        size_t group_comp_bytes_local = 0;
        size_t max_src_bytes_per_chunk = 0;

        for (int q_val = 0; q_val < 256; ++q_val) {
            size_t size1 = permuted_vec1[q_val].size();
            size_t size2 = permuted_vec2[q_val].size();
            if (size1 > 0) max_src_bytes_per_chunk = max(max_src_bytes_per_chunk, min(size1, (size_t)HUF_COMPRESS_SRCSIZE_MAX));
            if (size2 > 0) max_src_bytes_per_chunk = max(max_src_bytes_per_chunk, min(size2, (size_t)HUF_COMPRESS_SRCSIZE_MAX));
        }

        if (max_src_bytes_per_chunk > 0) {
             try {
                 thread_buffers[tid].ensure_capacity(HUF_compressBound(max_src_bytes_per_chunk));
             } catch (const std::bad_alloc& e) {
                 #pragma omp critical (cerr)
                 cerr << "Error: Failed to resize thread buffer. GroupIdx: " << group_idx << ", MaxChunk: " << max_src_bytes_per_chunk << ". Error: " << e.what() << endl;
                 // Mark results for this group as error?
                 continue; // Skip this group
             }
         }

        for (int stream_idx = 0; stream_idx < 2; ++stream_idx) {
            for (int q_val = 0; q_val < 256; q_val++) {
                vector<uint8_t>& vec_to_compress = (stream_idx == 0) ? permuted_vec1[q_val] : permuted_vec2[q_val];
                size_t total_stream_orig_bytes = vec_to_compress.size();

                // Get references to the correct output slots
                uint8_t*& current_result_data_ptr = (stream_idx == 0) ? result_data_ptr1[group_idx][q_val] : result_data_ptr2[group_idx][q_val];
                size_t& current_result_metadata = (stream_idx == 0) ? result_metadata1[group_idx][q_val] : result_metadata2[group_idx][q_val];
                uint8_t& current_result_type = (stream_idx == 0) ? result_type1[group_idx][q_val] : result_type2[group_idx][q_val];
                size_t& current_result_original_size = (stream_idx == 0) ? result_original_size1[group_idx][q_val] : result_original_size2[group_idx][q_val];

                // Initialize outputs (Type already initialized to TYPE_ERROR in main loop)
                current_result_data_ptr = nullptr;
                current_result_metadata = 0;
                // current_result_type = TYPE_ERROR; // Already done
                current_result_original_size = total_stream_orig_bytes;

                if (total_stream_orig_bytes == 0) {
                    current_result_type = TYPE_UNCOMPRESSED; // Mark empty as uncompressed
                    continue;
                }

                group_orig_bytes_local += total_stream_orig_bytes;

                // --- Chunking Logic ---
                if (total_stream_orig_bytes > HUF_COMPRESS_SRCSIZE_MAX) {
                    current_result_type = TYPE_CHUNKED_HUF; // Set type to chunked
                    size_t num_sub_chunks = (total_stream_orig_bytes + HUF_COMPRESS_SRCSIZE_MAX - 1) / HUF_COMPRESS_SRCSIZE_MAX;
                    current_result_metadata = num_sub_chunks; // Store number of chunks

                    vector<ChunkResult>* sub_chunk_results_vec = nullptr;
                    try {
                         sub_chunk_results_vec = new vector<ChunkResult>(num_sub_chunks);
                    } catch (const std::bad_alloc& e) {
                         #pragma omp critical (cerr)
                         cerr << "Error: Failed to allocate ChunkResult vector. Group: " << group_idx << " qVal: " << q_val << " Stream: " << stream_idx << ". Error: " << e.what() << endl;
                         current_result_type = TYPE_ERROR; // Mark as error
                         continue; // Skip this q_val/stream
                    }
                    current_result_data_ptr = reinterpret_cast<uint8_t*>(sub_chunk_results_vec);

                    size_t stream_total_comp_bytes = 0;
                    size_t processed_bytes = 0;
                    bool sub_chunk_error = false;

                    for (size_t chunk_idx = 0; chunk_idx < num_sub_chunks; ++chunk_idx) {
                        if (sub_chunk_error) break;

                        ChunkResult& sub_chunk_res = (*sub_chunk_results_vec)[chunk_idx];
                        size_t current_sub_chunk_size = min((size_t)HUF_COMPRESS_SRCSIZE_MAX, total_stream_orig_bytes - processed_bytes);
                        const uint8_t* sub_chunk_src_ptr = vec_to_compress.data() + processed_bytes;

                        sub_chunk_res.original_size = current_sub_chunk_size;
                        sub_chunk_res.data = nullptr;
                        sub_chunk_res.compressed_size = 0;
                        sub_chunk_res.type = TYPE_ERROR;

                        if (current_sub_chunk_size <= 64) {
                             uint8_t* final_buffer = static_cast<uint8_t*>(malloc(current_sub_chunk_size));
                             if (final_buffer) {
                                 memcpy(final_buffer, sub_chunk_src_ptr, current_sub_chunk_size);
                                 sub_chunk_res.data = final_buffer;
                                 sub_chunk_res.compressed_size = current_sub_chunk_size;
                                 sub_chunk_res.type = TYPE_UNCOMPRESSED;
                                 stream_total_comp_bytes += current_sub_chunk_size;
                             } else {
                                  #pragma omp critical (cerr)
                                  cerr << "Error: malloc failed for small sub-chunk. Group: " << group_idx << " qVal: " << q_val << " Stream: " << stream_idx << " Chunk: " << chunk_idx << endl;
                                  sub_chunk_error = true;
                             }
                        } else {
                            vector<uint8_t>& thread_buffer = (stream_idx == 0) ? thread_buffers[tid].buffer1 : thread_buffers[tid].buffer2;
                            if (thread_buffer.empty()) {
                                #pragma omp critical (cerr)
                                cerr << "Error: Thread buffer empty during sub-chunk compression. Group: " << group_idx << " qVal: " << q_val << " Stream: " << stream_idx << " Chunk: " << chunk_idx << endl;
                                sub_chunk_error = true;
                            } else {
                                uint8_t* compress_buffer = thread_buffer.data();
                                size_t dst_capacity = thread_buffer.size(); // Use actual size, ensure_capacity was called
                                size_t c_size = HUF_compress(compress_buffer, dst_capacity, sub_chunk_src_ptr, current_sub_chunk_size);

                                if (HUF_isError(c_size)) {
                                    #pragma omp critical (cerr)
                                    cerr << "Warning: HUF_compress failed for sub-chunk. Group: " << group_idx << " qVal: " << q_val << " Stream: " << stream_idx << " Chunk: " << chunk_idx << ". Error: " << HUF_getErrorName(c_size) << ". Storing uncompressed." << endl;
                                    c_size = 0;
                                }

                                if (c_size == 0 || c_size >= current_sub_chunk_size * 0.99) { // Store uncompressed
                                    uint8_t* final_buffer = static_cast<uint8_t*>(malloc(current_sub_chunk_size));
                                    if (final_buffer) {
                                        memcpy(final_buffer, sub_chunk_src_ptr, current_sub_chunk_size);
                                        sub_chunk_res.data = final_buffer;
                                        sub_chunk_res.compressed_size = current_sub_chunk_size;
                                        sub_chunk_res.type = TYPE_UNCOMPRESSED;
                                        stream_total_comp_bytes += current_sub_chunk_size;
                                    } else {
                                        #pragma omp critical (cerr)
                                        cerr << "Error: malloc failed for uncompressed sub-chunk. Group: " << group_idx << " qVal: " << q_val << " Stream: " << stream_idx << " Chunk: " << chunk_idx << endl;
                                        sub_chunk_error = true;
                                    }
                                } else { // Store compressed
                                    uint8_t* final_buffer = static_cast<uint8_t*>(malloc(c_size));
                                    if (final_buffer) {
                                        memcpy(final_buffer, compress_buffer, c_size);
                                        sub_chunk_res.data = final_buffer;
                                        sub_chunk_res.compressed_size = c_size;
                                        sub_chunk_res.type = TYPE_COMPRESSED_HUF;
                                        stream_total_comp_bytes += c_size;
                                    } else {
                                        #pragma omp critical (cerr)
                                        cerr << "Error: malloc failed for compressed sub-chunk. Group: " << group_idx << " qVal: " << q_val << " Stream: " << stream_idx << " Chunk: " << chunk_idx << endl;
                                        sub_chunk_error = true;
                                    }
                                }
                            } // end else (thread_buffer not empty)
                        } // end else (sub_chunk_size > 64)
                        processed_bytes += current_sub_chunk_size;
                    } // end loop over sub-chunks

                    if (sub_chunk_error) {
                        if (sub_chunk_results_vec) {
                             for(auto& res : *sub_chunk_results_vec) { if(res.data) free(res.data); }
                             delete sub_chunk_results_vec;
                        }
                        current_result_data_ptr = nullptr;
                        current_result_metadata = 0;
                        current_result_type = TYPE_ERROR;
                        // Don't add to group_comp_bytes_local
                    } else {
                        group_comp_bytes_local += stream_total_comp_bytes;
                    }

                } else { // --- No Chunking Needed (Original Logic) ---
                    if (total_stream_orig_bytes <= 2) {
                        uint8_t* final_dst_buffer = static_cast<uint8_t*>(malloc(total_stream_orig_bytes));
                        if (!final_dst_buffer) {
                            #pragma omp critical (cerr)
                            cerr << "Error: malloc failed for small uncompressed buffer (Stream " << stream_idx + 1 << "). GroupIdx: " << group_idx << ", qVal: " << q_val << endl;
                            current_result_type = TYPE_ERROR;
                        } else {
                            memcpy(final_dst_buffer, vec_to_compress.data(), total_stream_orig_bytes);
                            current_result_data_ptr = final_dst_buffer;
                            current_result_metadata = total_stream_orig_bytes; // Store compressed_size
                            current_result_type = TYPE_UNCOMPRESSED;
                            group_comp_bytes_local += total_stream_orig_bytes;
                        }
                    } else { // Attempt compression
                        vector<uint8_t>& thread_buffer = (stream_idx == 0) ? thread_buffers[tid].buffer1 : thread_buffers[tid].buffer2;
                        if (thread_buffer.empty()) {
                             #pragma omp critical (cerr)
                             cerr << "Error: Thread buffer empty for non-chunked compression. GroupIdx: " << group_idx << ", qVal: " << q_val << " Stream: " << stream_idx << endl;
                             current_result_type = TYPE_ERROR;
                        } else {
                            uint8_t* compress_buffer = thread_buffer.data();
                            size_t dst_capacity = thread_buffer.size();
                            size_t c_size = HUF_compress(compress_buffer, dst_capacity, vec_to_compress.data(), total_stream_orig_bytes);

                            if (HUF_isError(c_size)) {
                                #pragma omp critical (cerr)
                                cerr << "Warning: HUF_compress failed (Stream " << stream_idx + 1 << "). GroupIdx: " << group_idx << ", qVal: " << q_val << ". Error: " << HUF_getErrorName(c_size) << ". Storing uncompressed." << endl;
                                c_size = 0;
                            }

                            if (c_size == 0 || c_size >= total_stream_orig_bytes * 0.99) { // Store uncompressed
                                uint8_t* final_dst_buffer = static_cast<uint8_t*>(malloc(total_stream_orig_bytes));
                                if (!final_dst_buffer) {
                                    #pragma omp critical (cerr)
                                    cerr << "Error: malloc failed for uncompressed buffer (Stream " << stream_idx + 1 << "). GroupIdx: " << group_idx << ", qVal: " << q_val << endl;
                                    current_result_type = TYPE_ERROR;
                                } else {
                                    memcpy(final_dst_buffer, vec_to_compress.data(), total_stream_orig_bytes);
                                    current_result_data_ptr = final_dst_buffer;
                                    current_result_metadata = total_stream_orig_bytes;
                                    current_result_type = TYPE_UNCOMPRESSED;
                                    group_comp_bytes_local += total_stream_orig_bytes;
                                }
                            } else { // Store compressed
                                uint8_t* final_dst_buffer = static_cast<uint8_t*>(malloc(c_size));
                                if (!final_dst_buffer) {
                                     #pragma omp critical (cerr)
                                     cerr << "Error: malloc failed for compressed buffer (Stream " << stream_idx + 1 << "). GroupIdx: " << group_idx << ", qVal: " << q_val << endl;
                                     current_result_type = TYPE_ERROR;
                                } else {
                                    memcpy(final_dst_buffer, compress_buffer, c_size);
                                    current_result_data_ptr = final_dst_buffer;
                                    current_result_metadata = c_size;
                                    current_result_type = TYPE_COMPRESSED_HUF;
                                    group_comp_bytes_local += c_size;
                                }
                            }
                        } // end else (thread buffer not empty)
                    } // end else (attempt compression)
                } // --- End Chunking Logic ---
            } // End q_val loop
        } // End stream_idx loop
        // --- End Compression Phase ---

        // Add this group's contribution to the total (reduction handles this automatically)
        // Note: group_comp_bytes_local is only incremented on success (type != TYPE_ERROR)
        total_orig_tensor_size_local += group_orig_bytes_local;
        total_comp_tensor_size_local += group_comp_bytes_local;

    } // End of parallel loop over scale groups

    // Assign reduced values back to output references
    total_orig_bytes_out = total_orig_tensor_size_local;
    total_comp_bytes_out = total_comp_tensor_size_local;
}

uint64_t serialize_quantized_tensor(
    std::ofstream& outFile,
    const std::string& tensor_name,
    const std::vector<ChunkResult>& chunk_results,
    const std::vector<size_t>& tensor_shape
) {
    uint64_t bytes_written = 0;
    if (!outFile.is_open() || !outFile) {
        cerr << "Error: Output file stream is not open or in a bad state for tensor " << tensor_name << endl;
        return 0;
    }

    // Calculate total elements from shape for empty check
    size_t total_elements = 1;
    if (tensor_shape.empty()) {
        total_elements = 0;
    } else {
        for(size_t dim : tensor_shape) {
            if (dim == 0) { total_elements = 0; break; }
            if (total_elements > SIZE_MAX / dim) { // Basic overflow check
                cerr << "Error: Potential overflow calculating total elements for tensor " << tensor_name << " during serialization." << endl;
                return 0; // Indicate error
            }
            total_elements *= dim;
        }
    }

    // Handle empty tensor case (write only header)
    if (chunk_results.empty()) {
        if (total_elements > 0) {
             cerr << "Error: No chunk results for non-empty tensor " << tensor_name << " during serialization." << endl;
             return 0; // Indicate error
        }
        // Write empty header
        uint32_t num_chunks = 0;
        outFile.write(reinterpret_cast<const char*>(&num_chunks), sizeof(uint32_t));
        bytes_written += sizeof(uint32_t);

        uint8_t num_dims = static_cast<uint8_t>(tensor_shape.size());
        outFile.write(reinterpret_cast<const char*>(&num_dims), sizeof(uint8_t));
        bytes_written += sizeof(uint8_t);

        for(size_t dim : tensor_shape) {
            uint32_t dim_32 = static_cast<uint32_t>(dim);
            outFile.write(reinterpret_cast<const char*>(&dim_32), sizeof(uint32_t));
            bytes_written += sizeof(uint32_t);
        }
        return bytes_written; // Return bytes written for empty header
    }

    // --- Write Header for non-empty tensor ---
    uint32_t num_chunks = static_cast<uint32_t>(chunk_results.size());
    outFile.write(reinterpret_cast<const char*>(&num_chunks), sizeof(uint32_t));
    bytes_written += sizeof(uint32_t);

    uint8_t num_dims = static_cast<uint8_t>(tensor_shape.size());
    outFile.write(reinterpret_cast<const char*>(&num_dims), sizeof(uint8_t));
    bytes_written += sizeof(uint8_t);

    for(size_t dim : tensor_shape) {
        uint32_t dim_32 = static_cast<uint32_t>(dim);
        outFile.write(reinterpret_cast<const char*>(&dim_32), sizeof(uint32_t));
        bytes_written += sizeof(uint32_t);
    }

    // --- Write Chunk Metadata Section ---
    for (const auto& chunk_res : chunk_results) {
        if (chunk_res.type == TYPE_ERROR) {
            cerr << "Error: Attempting to serialize chunk with error state for tensor " << tensor_name << "." << endl;
            return 0; // Indicate error
        }
        if (chunk_res.type > TYPE_COMPRESSED_HUF) { // Only 0 or 1 allowed
            cerr << "Error: Invalid type (" << (int)chunk_res.type << ") found in quantized chunk metadata for tensor " << tensor_name << "." << endl;
            return 0; // Indicate error
        }

        uint8_t type_u8 = chunk_res.type;
        uint32_t orig_s = static_cast<uint32_t>(chunk_res.original_size);
        uint32_t comp_s = static_cast<uint32_t>(chunk_res.compressed_size);

        outFile.write(reinterpret_cast<const char*>(&type_u8), sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char*>(&orig_s), sizeof(uint32_t));
        outFile.write(reinterpret_cast<const char*>(&comp_s), sizeof(uint32_t));
        bytes_written += sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint32_t);
    }

    // --- Write Data Payloads ---
    for (const auto& chunk_res : chunk_results) {
        if (chunk_res.compressed_size > 0) {
            if (chunk_res.data != nullptr) {
                outFile.write(reinterpret_cast<const char*>(chunk_res.data), chunk_res.compressed_size);
                bytes_written += chunk_res.compressed_size;
            } else {
                cerr << "Error: Null data pointer for non-zero size chunk during data serialization for tensor " << tensor_name << "." << endl;
                return 0; // Indicate error
            }
        }
    }

    if (!outFile) {
        cerr << "Error: Output file stream is in a bad state after writing tensor " << tensor_name << endl;
        return 0; // Indicate error
    }

    return bytes_written;
}

uint64_t serialize_original_tensor(
    std::ofstream& outFile,
    const std::string& tensor_name,
    const std::vector<std::vector<uint8_t*>>& result_data_ptr1,
    const std::vector<std::vector<size_t>>& result_metadata1,
    const std::vector<std::vector<uint8_t*>>& result_data_ptr2,
    const std::vector<std::vector<size_t>>& result_metadata2,
    const std::vector<std::vector<uint8_t>>& result_type1,
    const std::vector<std::vector<uint8_t>>& result_type2,
    const std::vector<std::vector<size_t>>& result_original_size1,
    const std::vector<std::vector<size_t>>& result_original_size2,
    const std::vector<uint32_t>& row_to_group_idx_map,
    size_t num_groups_size_t,
    const std::vector<size_t>& tensor_shape
) {
    uint64_t bytes_written = 0;
    if (!outFile.is_open() || !outFile) {
        cerr << "Error: Output file stream is not open or in a bad state for original tensor " << tensor_name << endl;
        return 0;
    }

    // --- Write File Header ---
    uint32_t num_groups = static_cast<uint32_t>(num_groups_size_t);
    outFile.write(reinterpret_cast<const char*>(&num_groups), sizeof(uint32_t));
    bytes_written += sizeof(uint32_t);

    uint8_t num_dims = static_cast<uint8_t>(tensor_shape.size());
    outFile.write(reinterpret_cast<const char*>(&num_dims), sizeof(uint8_t));
    bytes_written += sizeof(uint8_t);

    for(size_t dim : tensor_shape) {
        uint32_t dim_32 = static_cast<uint32_t>(dim);
        outFile.write(reinterpret_cast<const char*>(&dim_32), sizeof(uint32_t));
        bytes_written += sizeof(uint32_t);
    }

    uint32_t num_rows = (tensor_shape.empty() || tensor_shape.size() < 1) ? 0 : static_cast<uint32_t>(tensor_shape[0]);
    if (row_to_group_idx_map.size() != num_rows && num_rows > 0) {
        cerr << "Error serializing " << tensor_name << ": Mismatch between tensor shape rows ("
             << num_rows << ") and row_to_group_idx_map size (" << row_to_group_idx_map.size() << ")." << endl;
        return 0; // Indicate error
    }
    outFile.write(reinterpret_cast<const char*>(&num_rows), sizeof(uint32_t));
    bytes_written += sizeof(uint32_t);

    if (num_rows > 0) {
        outFile.write(reinterpret_cast<const char*>(row_to_group_idx_map.data()), num_rows * sizeof(uint32_t));
        bytes_written += num_rows * sizeof(uint32_t);
    }
    // --- End File Header ---

    // --- Write Data Payload ---
    for(uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        // Basic bounds check
        if (group_idx >= result_type1.size() || group_idx >= result_type2.size() ||
            result_type1[group_idx].size() != 256 || result_type2[group_idx].size() != 256) {
            cerr << "Error serializing " << tensor_name << ": Result vector size mismatch for group_idx " << group_idx << "." << endl;
            return 0; // Indicate error
        }

        for (int q_val = 0; q_val < 256; ++q_val) {
            // --- Write Stream 1 ---
            uint8_t type1 = result_type1[group_idx][q_val];
            uint32_t orig_size1_u32 = static_cast<uint32_t>(result_original_size1[group_idx][q_val]);
            uint32_t metadata1_u32 = static_cast<uint32_t>(result_metadata1[group_idx][q_val]);
            uint8_t* data_ptr1 = result_data_ptr1[group_idx][q_val];

            outFile.write(reinterpret_cast<const char*>(&orig_size1_u32), sizeof(uint32_t));
            outFile.write(reinterpret_cast<const char*>(&type1), sizeof(uint8_t));
            outFile.write(reinterpret_cast<const char*>(&metadata1_u32), sizeof(uint32_t));
            bytes_written += sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint32_t);

            if (type1 == TYPE_CHUNKED_HUF) {
                uint32_t num_sub_chunks = metadata1_u32;
                if (data_ptr1 == nullptr) {
                     cerr << "Error: Null data pointer for chunked stream (Stream 1). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                     return 0;
                 }
                vector<ChunkResult>* sub_chunk_vec_ptr = reinterpret_cast<vector<ChunkResult>*>(data_ptr1);
                if (sub_chunk_vec_ptr->size() != num_sub_chunks) {
                    cerr << "Error: Mismatch between stored chunk count and vector size (Stream 1). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                    return 0;
                }

                for (const auto& sub_chunk : *sub_chunk_vec_ptr) {
                    if (sub_chunk.type == TYPE_ERROR || (sub_chunk.compressed_size > 0 && sub_chunk.data == nullptr)) {
                        cerr << "Error: Invalid sub-chunk found during serialization (Stream 1). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                        return 0;
                    }
                    uint8_t sub_type = sub_chunk.type;
                    uint32_t sub_orig_size = static_cast<uint32_t>(sub_chunk.original_size);
                    uint32_t sub_comp_size = static_cast<uint32_t>(sub_chunk.compressed_size);

                    outFile.write(reinterpret_cast<const char*>(&sub_type), sizeof(uint8_t));
                    outFile.write(reinterpret_cast<const char*>(&sub_orig_size), sizeof(uint32_t));
                    outFile.write(reinterpret_cast<const char*>(&sub_comp_size), sizeof(uint32_t));
                    bytes_written += sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint32_t);
                    if (sub_comp_size > 0) {
                        outFile.write(reinterpret_cast<const char*>(sub_chunk.data), sub_comp_size);
                        bytes_written += sub_comp_size;
                    }
                 }
            } else if (type1 == TYPE_UNCOMPRESSED || type1 == TYPE_COMPRESSED_HUF) {
                size_t size_to_write = metadata1_u32;
                 if (size_to_write > 0) {
                     if (data_ptr1 == nullptr) {
                         cerr << "Error: Null data pointer for non-chunked stream (Stream 1, Type=" << (int)type1 << "). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                         return 0;
                     }
                     outFile.write(reinterpret_cast<const char*>(data_ptr1), size_to_write);
                     bytes_written += size_to_write;
                 }
            } else if (type1 == TYPE_ERROR) {
                 // Allow serialization even if some streams had errors during compression
                 // The error type is written, but no payload follows.
                 if (orig_size1_u32 > 0) {
                     // cerr << "Warning: Serializing TYPE_ERROR for non-empty stream (Stream 1). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << endl;
                 }
            } else {
                 cerr << "Error: Unexpected type (" << (int)type1 << ") encountered during serialization (Stream 1). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                 return 0;
            }

            // --- Write Stream 2 (Identical Logic) ---
            uint8_t type2 = result_type2[group_idx][q_val];
            uint32_t orig_size2_u32 = static_cast<uint32_t>(result_original_size2[group_idx][q_val]);
            uint32_t metadata2_u32 = static_cast<uint32_t>(result_metadata2[group_idx][q_val]);
            uint8_t* data_ptr2 = result_data_ptr2[group_idx][q_val];

            outFile.write(reinterpret_cast<const char*>(&orig_size2_u32), sizeof(uint32_t));
            outFile.write(reinterpret_cast<const char*>(&type2), sizeof(uint8_t));
            outFile.write(reinterpret_cast<const char*>(&metadata2_u32), sizeof(uint32_t));
            bytes_written += sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint32_t);

            if (type2 == TYPE_CHUNKED_HUF) {
                uint32_t num_sub_chunks = metadata2_u32;
                 if (data_ptr2 == nullptr) {
                     cerr << "Error: Null data pointer for chunked stream (Stream 2). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                     return 0;
                 }
                vector<ChunkResult>* sub_chunk_vec_ptr = reinterpret_cast<vector<ChunkResult>*>(data_ptr2);
                if (sub_chunk_vec_ptr->size() != num_sub_chunks) {
                     cerr << "Error: Mismatch between stored chunk count and vector size (Stream 2). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                     return 0;
                 }

                for (const auto& sub_chunk : *sub_chunk_vec_ptr) {
                    if (sub_chunk.type == TYPE_ERROR || (sub_chunk.compressed_size > 0 && sub_chunk.data == nullptr)) {
                         cerr << "Error: Invalid sub-chunk found during serialization (Stream 2). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                         return 0;
                     }
                     uint8_t sub_type = sub_chunk.type;
                     uint32_t sub_orig_size = static_cast<uint32_t>(sub_chunk.original_size);
                     uint32_t sub_comp_size = static_cast<uint32_t>(sub_chunk.compressed_size);

                     outFile.write(reinterpret_cast<const char*>(&sub_type), sizeof(uint8_t));
                     outFile.write(reinterpret_cast<const char*>(&sub_orig_size), sizeof(uint32_t));
                     outFile.write(reinterpret_cast<const char*>(&sub_comp_size), sizeof(uint32_t));
                     bytes_written += sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint32_t);
                     if (sub_comp_size > 0) {
                         outFile.write(reinterpret_cast<const char*>(sub_chunk.data), sub_comp_size);
                         bytes_written += sub_comp_size;
                     }
                 }
            } else if (type2 == TYPE_UNCOMPRESSED || type2 == TYPE_COMPRESSED_HUF) {
                size_t size_to_write = metadata2_u32;
                 if (size_to_write > 0) {
                      if (data_ptr2 == nullptr) {
                         cerr << "Error: Null data pointer for non-chunked stream (Stream 2, Type=" << (int)type2 << "). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                         return 0;
                     }
                     outFile.write(reinterpret_cast<const char*>(data_ptr2), size_to_write);
                     bytes_written += size_to_write;
                 }
            } else if (type2 == TYPE_ERROR) {
                 if (orig_size2_u32 > 0) {
                     // cerr << "Warning: Serializing TYPE_ERROR for non-empty stream (Stream 2). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << endl;
                 }
            } else {
                 cerr << "Error: Unexpected type (" << (int)type2 << ") encountered during serialization (Stream 2). Tensor: " << tensor_name << " Group: " << group_idx << " qVal: " << q_val << "." << endl;
                 return 0;
            }
        } // end q_val loop
    } // end group_idx loop
    // --- End Data Payload ---

    if (!outFile) {
        cerr << "Error: Output file stream is in a bad state after writing original tensor " << tensor_name << endl;
        return 0; // Indicate error
    }

    return bytes_written;
}