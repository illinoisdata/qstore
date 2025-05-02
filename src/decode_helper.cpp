#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <atomic>
#include <stdexcept>

// OpenMP for parallelization
#include <omp.h>

// External compression libraries
extern "C" {
    #include "huf.h"
}

#include "compression.h"

using namespace std;

#include <vector>
#include <cstdint> // For uint8_t etc.
#include <cstdlib> // For free

#define MAX_THREADS_QUANTIZED_DECOMPRESSION 48
#define MAX_THREADS_ORIG_DECOMPRESSION 48

// Helper struct matching the write format (can be defined locally if not in compression.h)
struct ChunkReadMetadata {
    uint8_t type;
    uint32_t original_size;
    uint32_t compressed_size;
    size_t data_payload_offset; // Offset in the input buffer where this chunk's data starts
    size_t output_buffer_offset; // Offset in the final output vector where this chunk belongs
};

CompressedDataBuffer read_segment_to_buffer(
    std::ifstream& inFile, // Pass open file stream
    const std::string& name,
    uint64_t offset,
    uint64_t size
) {
    CompressedDataBuffer data;
    data.tensor_name = name;

    // if (!inFile.is_open() || !inFile) {
    //     data.error_msg = "Input file stream is not open or in a bad state for tensor " + name;
    //     return data;
    // }

    // Check if size is zero (valid case for empty tensors)
    if (size == 0) {
        data.success = true;
        return data;
    }

    // Seek to the specified offset
    inFile.seekg(offset, ios::beg);
    // if (!inFile) {
    //     // Clear error flags before setting message
    //     inFile.clear();
    //     data.error_msg = "Failed to seek to offset " + to_string(offset) + " for tensor " + name;
    //     return data;
    // }

    // Resize buffer and read the segment
    try {
        data.buffer.resize(size);
    } catch (const std::bad_alloc& e) {
        data.error_msg = "Failed to allocate buffer of size " + to_string(size) + " for tensor " + name + ": " + e.what();
        return data;
    }

    inFile.read(reinterpret_cast<char*>(data.buffer.data()), size);
    // if (!inFile) {
    //     // Clear error flags before setting message
    //     inFile.clear();
    //     // Check how many bytes were actually read
    //     streamsize bytes_read = inFile.gcount();
    //     data.error_msg = "Failed to read " + to_string(size) + " bytes (read " + to_string(bytes_read) + ") from offset " + to_string(offset) + " for tensor " + name;
    //     data.buffer.clear(); // Clear potentially partial read
    //     return data;
    // }

    data.success = true;
    return data;
}

vector<uint8_t> decompress_quantized_tensor_chunked(
    const vector<uint8_t>& compressed_buffer_vec, // Input buffer (contains ONE tensor segment)
    const string& tensor_name // For error messages
) {
    if (compressed_buffer_vec.empty()) {
        // Check header consistency even for empty buffer?
        // The serialization writes a header even for empty tensors.
        // Let's assume the read_segment logic handles truly empty segments correctly.
        // If the segment isn't empty but represents an empty tensor, the header read will handle it.
        // return {}; // Empty input -> empty output (Potentially incorrect if header exists)
    }

    const char* buffer_ptr = reinterpret_cast<const char*>(compressed_buffer_vec.data());
    size_t buffer_size = compressed_buffer_vec.size();
    size_t current_offset = 0;

    auto read_from_buffer = [&](void* dest, size_t bytes_to_read) {
        if (current_offset + bytes_to_read > buffer_size) {
            throw runtime_error("[Tensor: " + tensor_name + "] Read past end of buffer (quantized header/metadata). Offset: " + to_string(current_offset) + ", Reading: " + to_string(bytes_to_read) + ", Size: " + to_string(buffer_size));
        }
        memcpy(dest, buffer_ptr + current_offset, bytes_to_read);
        current_offset += bytes_to_read;
    };

    // --- Read Header Sequentially ---
    uint32_t num_chunks = 0;
    // Check if buffer is large enough for num_chunks
    if (buffer_size < sizeof(uint32_t)) {
         throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (num_chunks). Size: " + to_string(buffer_size));
    }
    read_from_buffer(&num_chunks, sizeof(uint32_t));

    uint8_t num_dims = 0;
    // Check if buffer is large enough for num_dims
    if (buffer_size < current_offset + sizeof(uint8_t)) {
         throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (num_dims). Size: " + to_string(buffer_size));
    }
    read_from_buffer(&num_dims, sizeof(uint8_t));

    vector<size_t> tensor_shape(num_dims);
    size_t total_elements_header = (num_dims > 0) ? 1 : 0;
    // Check if buffer is large enough for dimensions
    if (buffer_size < current_offset + num_dims * sizeof(uint32_t)) {
         throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (dimensions). Size: " + to_string(buffer_size));
    }
    for (uint8_t i = 0; i < num_dims; ++i) {
        uint32_t dim_u32;
        read_from_buffer(&dim_u32, sizeof(uint32_t));
        tensor_shape[i] = static_cast<size_t>(dim_u32);
        if (dim_u32 == 0 && num_dims > 0) { // Allow shape=[] for scalar/empty
             throw runtime_error("[Tensor: " + tensor_name + "] Tensor dimension " + to_string(i) + " is zero.");
        }
        // Basic overflow check during multiplication
        if (total_elements_header > 0 && dim_u32 > 0 && total_elements_header > SIZE_MAX / dim_u32) {
             throw runtime_error("[Tensor: " + tensor_name + "] Overflow calculating total elements from header shape.");
        }
        if (dim_u32 > 0) {
            total_elements_header *= dim_u32;
        } else if (num_dims > 0) {
             total_elements_header = 0; // If any dim is 0, total elements is 0
        }
    }

    // Handle case where num_chunks is 0 (empty tensor)
    if (num_chunks == 0) {
         if (total_elements_header != 0) {
             throw runtime_error("[Tensor: " + tensor_name + "] Inconsistent header: num_chunks is 0 but shape implies non-zero elements (" + to_string(total_elements_header) + ").");
         }
         // Check if buffer size matches header size
         if (current_offset != buffer_size) {
              throw runtime_error("[Tensor: " + tensor_name + "] Buffer size (" + to_string(buffer_size) + ") does not match header size (" + to_string(current_offset) + ") for empty tensor.");
         }
         return {}; // 0 chunks means empty tensor
     }


    // --- Read Chunk Metadata Sequentially ---
    vector<ChunkReadMetadata> chunk_metadata(num_chunks);
    size_t total_original_size = 0;
    size_t metadata_section_size = num_chunks * (sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint32_t));
    // Check if buffer is large enough for metadata section
    if (buffer_size < current_offset + metadata_section_size) {
         throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for chunk metadata. Size: " + to_string(buffer_size) + ", Needed: " + to_string(current_offset + metadata_section_size));
    }

    for (uint32_t i = 0; i < num_chunks; ++i) {
        read_from_buffer(&chunk_metadata[i].type, sizeof(uint8_t));
        read_from_buffer(&chunk_metadata[i].original_size, sizeof(uint32_t));
        read_from_buffer(&chunk_metadata[i].compressed_size, sizeof(uint32_t));
        total_original_size += chunk_metadata[i].original_size;

        // Validation
        if (chunk_metadata[i].type > 1) { // Only 0 (uncomp) or 1 (comp) allowed
             throw runtime_error("[Tensor: " + tensor_name + "] Invalid chunk type (" + to_string(chunk_metadata[i].type) + ") encountered, chunk " + to_string(i));
        }
        if (chunk_metadata[i].type == 0 && chunk_metadata[i].original_size != chunk_metadata[i].compressed_size) {
              throw runtime_error("[Tensor: " + tensor_name + "] Uncompressed chunk size mismatch, chunk " + to_string(i) + ". Original=" + to_string(chunk_metadata[i].original_size) + ", Stored=" + to_string(chunk_metadata[i].compressed_size));
         }
    }

    // Validate total original size against shape
    if (total_original_size != total_elements_header) {
         throw runtime_error("[Tensor: " + tensor_name + "] Total original size from chunks (" + to_string(total_original_size)
              + ") does not match size calculated from header shape (" + to_string(total_elements_header) + ")");
    }

    // --- Calculate Payload and Output Offsets Sequentially ---
    size_t current_payload_offset_tracker = current_offset; // Data starts after all metadata
    size_t current_output_offset_tracker = 0;
    for (uint32_t i = 0; i < num_chunks; ++i) {
        chunk_metadata[i].data_payload_offset = current_payload_offset_tracker;
        chunk_metadata[i].output_buffer_offset = current_output_offset_tracker;
        current_payload_offset_tracker += chunk_metadata[i].compressed_size;
        current_output_offset_tracker += chunk_metadata[i].original_size;
    }

    // Check if the calculated end of payload matches buffer size
    if (current_payload_offset_tracker != buffer_size) {
        throw runtime_error("[Tensor: " + tensor_name + "] Calculated payload end offset (" + to_string(current_payload_offset_tracker)
             + ") does not match buffer size (" + to_string(buffer_size) + ")");
    }
    // --- End Sequential Setup ---


    // --- Allocate Output Buffer ---
    vector<uint8_t> output_data;
    try {
        // Resize only if total_original_size > 0 to avoid issues with empty tensors
        if (total_original_size > 0) {
            output_data.resize(total_original_size);
        } else {
            // If total_original_size is 0, we should have returned earlier (num_chunks == 0)
            // This is a safeguard.
            return {};
        }
    } catch (const std::bad_alloc& e) {
        throw runtime_error("[Tensor: " + tensor_name + "] Failed to allocate output buffer: " + e.what());
    }

    // --- Decompress Data Payloads in Parallel ---
    atomic<bool> decompression_error_occurred(false); // Flag for errors in parallel region

    #pragma omp parallel for schedule(dynamic) num_threads(MAX_THREADS_QUANTIZED_DECOMPRESSION) shared(decompression_error_occurred, chunk_metadata, buffer_ptr, output_data, tensor_name, buffer_size, total_original_size)
    for (uint32_t i = 0; i < num_chunks; ++i) {
        if (decompression_error_occurred.load(std::memory_order_relaxed)) {
             continue; // Skip processing if an error occurred elsewhere
        }

        const auto& meta = chunk_metadata[i];

        if (meta.original_size == 0) {
            if (meta.compressed_size != 0) {
                 #pragma omp critical (cerr_quant_decomp)
                 cerr << "Warning: Zero original size but non-zero compressed size for tensor " << tensor_name << ", chunk " << i << endl;
            }
            continue; // Skip empty chunks
        }

        // Bounds check for output buffer access
        if (meta.output_buffer_offset + meta.original_size > total_original_size) {
             #pragma omp critical (cerr_quant_decomp)
             cerr << "ERROR: Output buffer overflow detected for tensor " << tensor_name << ", chunk " << i << endl;
             decompression_error_occurred.store(true);
             continue;
        }
        // Bounds check for input buffer access
        if (meta.data_payload_offset + meta.compressed_size > buffer_size) {
             #pragma omp critical (cerr_quant_decomp)
             cerr << "ERROR: Input buffer overflow detected reading payload for tensor " << tensor_name << ", chunk " << i << endl;
             decompression_error_occurred.store(true);
             continue;
        }


        const char* chunk_data_ptr = buffer_ptr + meta.data_payload_offset;
        uint8_t* output_chunk_ptr = output_data.data() + meta.output_buffer_offset;

        if (meta.type == 0) { // Uncompressed
            memcpy(output_chunk_ptr, chunk_data_ptr, meta.original_size);
        } else if (meta.type == 1) { // Compressed
            size_t result = HUF_decompress(output_chunk_ptr, meta.original_size,
                                            chunk_data_ptr, meta.compressed_size);

            if (HUF_isError(result)) {
                #pragma omp critical (cerr_quant_decomp)
                cerr << "ERROR: HUF_decompress failed for tensor " << tensor_name
                        << ", chunk " << i << ": " << HUF_getErrorName(result) << endl;
                decompression_error_occurred.store(true);
                continue; // Skip to next chunk on error
            }
            // Size check is implicitly done by HUF_decompress returning the size or an error
            if (result != meta.original_size) {
                 #pragma omp critical (cerr_quant_decomp)
                 cerr << "ERROR: HUF_decompress size mismatch for tensor " << tensor_name << ", chunk " << i << ". Expected=" << meta.original_size << ", Got=" << result << endl;
                 decompression_error_occurred.store(true);
                 continue;
            }
        }
        // No else needed, type validation happened earlier
    } // --- End Parallel For ---

    if (decompression_error_occurred.load()) {
         throw runtime_error("[Tensor: " + tensor_name + "] One or more errors occurred during parallel chunk decompression.");
    }

    return output_data;
}


vector<uint16_t> decompress_tensor_from_buffer(
    const vector<uint8_t>& compressed_buffer_vec, // Input buffer (contains ONE tensor segment)
    const string& tensor_name, // For error messages
    const vector<uint8_t>& decompressed_quantized_vec
) {
    // Handle empty buffer case - check header consistency
    if (compressed_buffer_vec.empty()) {
        // An empty segment should correspond to an empty tensor.
        // The header read logic below will validate consistency.
        // If aux data is not empty, that's an inconsistency caught later.
    }

    const char* buffer_ptr = reinterpret_cast<const char*>(compressed_buffer_vec.data());
    size_t buffer_size = compressed_buffer_vec.size();
    size_t current_offset = 0;

    auto read_from_buffer = [&](void* dest, size_t bytes_to_read) {
        if (current_offset + bytes_to_read > buffer_size) {
            throw runtime_error("[Tensor: " + tensor_name + "] Read past end of buffer (orig header/metadata). Offset: " + to_string(current_offset) + ", Reading: " + to_string(bytes_to_read) + ", Size: " + to_string(buffer_size));
        }
        memcpy(dest, buffer_ptr + current_offset, bytes_to_read);
        current_offset += bytes_to_read;
    };

    // --- Read File Header (from buffer) ---
    // 1. Number of scale groups
    uint32_t num_groups = 0;
    if (buffer_size < sizeof(uint32_t)) throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (num_groups). Size: " + to_string(buffer_size));
    read_from_buffer(&num_groups, sizeof(uint32_t));

    // 2. Tensor Shape
    uint8_t num_dims = 0;
    if (buffer_size < current_offset + sizeof(uint8_t)) throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (num_dims). Size: " + to_string(buffer_size));
    read_from_buffer(&num_dims, sizeof(uint8_t));
    // Allow 0 dimensions for empty/scalar tensors represented with num_groups=0
    if (num_dims != 2 && num_groups > 0) { // Only enforce 2D if groups exist
        throw runtime_error("[Tensor: " + tensor_name + "] Error: Expected tensor dimensions to be 2 for non-empty grouped tensor, but got " + to_string(num_dims));
    }

    vector<size_t> tensor_shape(num_dims);
    vector<uint32_t> tensor_shape_u32(num_dims);
    if (buffer_size < current_offset + num_dims * sizeof(uint32_t)) throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (dimensions). Size: " + to_string(buffer_size));
    size_t total_elements = (num_dims > 0) ? 1 : 0;
    for (uint8_t i = 0; i < num_dims; ++i) {
        read_from_buffer(&tensor_shape_u32[i], sizeof(uint32_t));
        tensor_shape[i] = static_cast<size_t>(tensor_shape_u32[i]);
        if (tensor_shape[i] == 0 && num_dims > 0) {
             // Allow zero dimensions only if num_groups is also 0 (checked later)
             if (num_groups > 0) throw runtime_error("[Tensor: " + tensor_name + "] Error: Tensor dimension " + to_string(i) + " is zero for non-empty grouped tensor.");
             total_elements = 0; // Ensure total_elements is 0 if any dim is 0
        }
        if (total_elements > 0 && tensor_shape[i] > 0 && total_elements > SIZE_MAX / tensor_shape[i]) {
             throw runtime_error("[Tensor: " + tensor_name + "] Overflow calculating total elements from header shape.");
        }
        if (tensor_shape[i] > 0) {
            total_elements *= tensor_shape[i];
        } else {
            total_elements = 0;
        }
    }
    size_t num_rows = (num_dims >= 1) ? tensor_shape[0] : 0;
    size_t num_cols = (num_dims >= 2) ? tensor_shape[1] : 0;

    // 3. Number of rows in the map
    uint32_t num_rows_in_map_header = 0;
    if (buffer_size < current_offset + sizeof(uint32_t)) throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (num_rows_in_map). Size: " + to_string(buffer_size));
    read_from_buffer(&num_rows_in_map_header, sizeof(uint32_t));
    if (num_rows_in_map_header != num_rows) {
         throw runtime_error("[Tensor: " + tensor_name + "] Error: Mismatch between tensor shape rows (" + to_string(num_rows) + ") and header row count (" + to_string(num_rows_in_map_header) + ")");
    }

    // 4. Row -> group idx map
    vector<uint32_t> loaded_row_to_group_idx(num_rows);
    size_t map_bytes = num_rows * sizeof(uint32_t);
    if (buffer_size < current_offset + map_bytes) throw runtime_error("[Tensor: " + tensor_name + "] Buffer too small for header (row_map). Size: " + to_string(buffer_size));
    if (num_rows > 0) {
        read_from_buffer(loaded_row_to_group_idx.data(), map_bytes);
    }
    // --- End File Header ---

    // Handle empty tensor case (identified by num_groups == 0)
    if (num_groups == 0) {
        if (total_elements != 0) {
             throw runtime_error("[Tensor: " + tensor_name + "] Inconsistent header: num_groups is 0 but shape implies non-zero elements (" + to_string(total_elements) + ").");
        }
        if (decompressed_quantized_vec.size() != 0) {
             throw runtime_error("[Tensor: " + tensor_name + "] Inconsistent data: num_groups is 0 but auxiliary quantized data is not empty.");
        }
        // Check if buffer size matches header size
        if (current_offset != buffer_size) {
             throw runtime_error("[Tensor: " + tensor_name + "] Buffer size (" + to_string(buffer_size) + ") does not match header size (" + to_string(current_offset) + ") for empty tensor.");
        }
        return {}; // Return empty vector for empty tensor
    }

    // Validate consistency between total_elements and auxiliary data size
    if (total_elements != decompressed_quantized_vec.size()) {
         throw runtime_error("[Tensor: " + tensor_name + "] Mismatch between total elements from shape (" + to_string(total_elements) + ") and auxiliary quantized data size (" + to_string(decompressed_quantized_vec.size()) + ").");
    }


    // --- Read Data Payload and Decompress (from buffer) ---
    vector<vector<vector<uint8_t>>> decompressed_stream1(num_groups, vector<vector<uint8_t>>(256));
    vector<vector<vector<uint8_t>>> decompressed_stream2(num_groups, vector<vector<uint8_t>>(256));

    atomic<bool> decompression_error_occurred(false); // Flag to signal errors

    // --- Sequential Metadata Read and Payload Decompression ---
    for (uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
        if (decompression_error_occurred.load()) break; // Check flag at outer loop

        for (int q_val = 0; q_val < 256; ++q_val) {
            if (decompression_error_occurred.load()) break; // Check flag at middle loop

            for (int stream_idx = 0; stream_idx < 2; ++stream_idx) {
                if (decompression_error_occurred.load()) break; // Check flag at inner loop

                // Read metadata for the current stream (sequentially)
                uint32_t orig_size_u32 = 0;
                uint8_t type = 0;
                uint32_t metadata_u32 = 0;
                try {
                    read_from_buffer(&orig_size_u32, sizeof(uint32_t));
                    read_from_buffer(&type, sizeof(uint8_t));
                    read_from_buffer(&metadata_u32, sizeof(uint32_t));
                } catch (const std::runtime_error& e) {
                     #pragma omp critical (cerr_orig_decomp)
                     cerr << "ERROR reading metadata for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << ": " << e.what() << endl;
                     decompression_error_occurred.store(true);
                     break; // Exit inner loop
                }

                size_t orig_size = static_cast<size_t>(orig_size_u32);
                vector<uint8_t>& target_stream = (stream_idx == 0) ?
                                                 decompressed_stream1[group_idx][q_val] :
                                                 decompressed_stream2[group_idx][q_val];

                if (orig_size == 0) {
                    target_stream.clear();
                    if (metadata_u32 != 0 && type != TYPE_ERROR) {
                         #pragma omp critical (cerr_orig_decomp)
                         cerr << "Warning: Zero original size but non-zero metadata/type for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << endl;
                    }
                    continue; // Skip to next stream
                }

                // Allocate target buffer
                try {
                    target_stream.resize(orig_size);
                } catch (const std::bad_alloc& e) {
                    #pragma omp critical (cerr_orig_decomp)
                    cerr << "ERROR: Failed to allocate target buffer for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << ": " << e.what() << endl;
                    decompression_error_occurred.store(true);
                    break; // Exit inner loop
                }

                // Process based on type
                if (type == TYPE_UNCOMPRESSED || type == TYPE_COMPRESSED_HUF) {
                    size_t comp_size = static_cast<size_t>(metadata_u32);
                    if (comp_size == 0) {
                         #pragma omp critical (cerr_orig_decomp)
                         cerr << "ERROR: Zero compressed size for non-empty stream (Type " << (int)type << ") tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << endl;
                         decompression_error_occurred.store(true);
                         break; // Exit inner loop
                    }
                    if (current_offset + comp_size > buffer_size) {
                         #pragma omp critical (cerr_orig_decomp)
                         cerr << "ERROR: Buffer overflow reading payload (Type " << (int)type << ") tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << endl;
                         decompression_error_occurred.store(true);
                         break; // Exit inner loop
                    }
                    const char* payload_ptr = buffer_ptr + current_offset;
                    current_offset += comp_size;

                    if (type == TYPE_UNCOMPRESSED) {
                        if (comp_size != orig_size) {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR: Uncompressed size mismatch for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << ". Original=" << orig_size << ", Stored=" << comp_size << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit inner loop
                        }
                        memcpy(target_stream.data(), payload_ptr, orig_size);
                    } else { // TYPE_COMPRESSED_HUF
                        size_t result = HUF_decompress(target_stream.data(), orig_size, payload_ptr, comp_size);
                        if (HUF_isError(result)) {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR: HUF_decompress failed for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << ": " << HUF_getErrorName(result) << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit inner loop
                        }
                        if (result != orig_size) {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR: HUF_decompress size mismatch for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << ". Expected=" << orig_size << ", Got=" << result << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit inner loop
                        }
                    }
                } else if (type == TYPE_CHUNKED_HUF) {
                    uint32_t num_sub_chunks = metadata_u32;
                    if (num_sub_chunks == 0) {
                         #pragma omp critical (cerr_orig_decomp)
                         cerr << "ERROR: Zero sub-chunks for chunked stream tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << endl;
                         decompression_error_occurred.store(true);
                         break; // Exit inner loop
                    }

                    size_t current_output_pos = 0;
                    size_t total_decompressed_size_check = 0;

                    for (uint32_t chunk_idx = 0; chunk_idx < num_sub_chunks; ++chunk_idx) {
                        if (decompression_error_occurred.load()) break; // Check flag within sub-chunk loop

                        uint8_t sub_type = 0;
                        uint32_t sub_orig_size_u32 = 0;
                        uint32_t sub_comp_size_u32 = 0;
                        try {
                            read_from_buffer(&sub_type, sizeof(uint8_t));
                            read_from_buffer(&sub_orig_size_u32, sizeof(uint32_t));
                            read_from_buffer(&sub_comp_size_u32, sizeof(uint32_t));
                        } catch (const std::runtime_error& e) {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR reading sub-chunk metadata for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << ": " << e.what() << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit sub-chunk loop
                        }
                        size_t sub_orig_size = static_cast<size_t>(sub_orig_size_u32);
                        size_t sub_comp_size = static_cast<size_t>(sub_comp_size_u32);
                        total_decompressed_size_check += sub_orig_size;

                        if (sub_orig_size == 0) {
                             if (sub_comp_size != 0) {
                                 #pragma omp critical (cerr_orig_decomp)
                                 cerr << "Warning: Zero sub-chunk original size but non-zero compressed size for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << endl;
                             }
                             continue; // Skip empty sub-chunk
                        }

                        if (current_output_pos + sub_orig_size > orig_size) {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR: Output buffer overflow processing sub-chunk for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit sub-chunk loop
                        }
                        if (current_offset + sub_comp_size > buffer_size) {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR: Input buffer overflow reading sub-chunk payload for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit sub-chunk loop
                        }

                        const char* sub_payload_ptr = buffer_ptr + current_offset;
                        current_offset += sub_comp_size;
                        uint8_t* sub_output_ptr = target_stream.data() + current_output_pos;

                        if (sub_type == TYPE_UNCOMPRESSED) {
                            if (sub_comp_size != sub_orig_size) {
                                 #pragma omp critical (cerr_orig_decomp)
                                 cerr << "ERROR: Uncompressed sub-chunk size mismatch for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << ". Original=" << sub_orig_size << ", Stored=" << sub_comp_size << endl;
                                 decompression_error_occurred.store(true);
                                 break; // Exit sub-chunk loop
                            }
                            memcpy(sub_output_ptr, sub_payload_ptr, sub_orig_size);
                        } else if (sub_type == TYPE_COMPRESSED_HUF) {
                            size_t result = HUF_decompress(sub_output_ptr, sub_orig_size, sub_payload_ptr, sub_comp_size);
                            if (HUF_isError(result)) {
                                 #pragma omp critical (cerr_orig_decomp)
                                 cerr << "ERROR: HUF_decompress failed for sub-chunk tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << ": " << HUF_getErrorName(result) << endl;
                                 decompression_error_occurred.store(true);
                                 break; // Exit sub-chunk loop
                            }
                            if (result != sub_orig_size) {
                                 #pragma omp critical (cerr_orig_decomp)
                                 cerr << "ERROR: HUF_decompress sub-chunk size mismatch for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << ". Expected=" << sub_orig_size << ", Got=" << result << endl;
                                 decompression_error_occurred.store(true);
                                 break; // Exit sub-chunk loop
                            }
                        } else {
                             #pragma omp critical (cerr_orig_decomp)
                             cerr << "ERROR: Invalid sub-chunk type (" << (int)sub_type << ") for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << " Chunk " << chunk_idx << endl;
                             decompression_error_occurred.store(true);
                             break; // Exit sub-chunk loop
                        }
                        current_output_pos += sub_orig_size;
                    } // End loop over sub-chunks

                    // Check flag after sub-chunk loop finishes or breaks
                    if (decompression_error_occurred.load()) break; // Exit inner loop if sub-chunk error occurred

                    if (total_decompressed_size_check != orig_size) {
                         #pragma omp critical (cerr_orig_decomp)
                         cerr << "ERROR: Total sub-chunk original size (" << total_decompressed_size_check << ") mismatch with stream original size (" << orig_size << ") for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << endl;
                         decompression_error_occurred.store(true);
                         break; // Exit inner loop
                    }

                } else if (type == TYPE_ERROR) {
                     #pragma omp critical (cerr_orig_decomp)
                     cerr << "Warning: Encountered error type (" << (int)type << ") during decompression for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << ". Output stream will be empty." << endl;
                     target_stream.clear();
                } else {
                     #pragma omp critical (cerr_orig_decomp)
                     cerr << "ERROR: Unknown stream type (" << (int)type << ") encountered for tensor " << tensor_name << " Group " << group_idx << " qVal " << q_val << " Stream " << stream_idx << endl;
                     decompression_error_occurred.store(true);
                     break; // Exit inner loop
                }
            } // end stream_idx loop
        } // end q_val loop
    } // end group_idx loop

    // Check if an error occurred anywhere in the loops
    if (decompression_error_occurred.load()) {
        throw runtime_error("[Tensor: " + tensor_name + "] One or more errors occurred during original stream decompression.");
    }

    // Check if we consumed the entire buffer
    if (current_offset != buffer_size) {
         throw runtime_error("[Tensor: " + tensor_name + "] Buffer size mismatch after reading payload. Expected offset " + to_string(buffer_size) + ", actual offset " + to_string(current_offset));
    }
    // --- End Data Payload Read ---


    // --- Reconstruct Original Tensor ---
    // This part remains sequential as it depends on the order of elements in decompressed_quantized_vec
    vector<uint16_t> output_orig_vec;
    try {
        output_orig_vec.resize(total_elements);
    } catch (const std::bad_alloc& e) {
        throw runtime_error("[Tensor: " + tensor_name + "] Failed to allocate final output buffer: " + e.what());
    }

    vector<vector<size_t>> stream1_iter(num_groups, vector<size_t>(256, 0));
    vector<vector<size_t>> stream2_iter(num_groups, vector<size_t>(256, 0));

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        if (row_idx >= loaded_row_to_group_idx.size()) throw runtime_error("[Tensor: " + tensor_name + "] Row index out of bounds accessing row map during reconstruction.");
        uint32_t group_idx_u32 = loaded_row_to_group_idx[row_idx];
        if (group_idx_u32 >= num_groups) throw runtime_error("[Tensor: " + tensor_name + "] Invalid group index " + to_string(group_idx_u32) + " in row map during reconstruction.");
        size_t group_idx = static_cast<size_t>(group_idx_u32);

        size_t start_index = row_idx * num_cols;
        size_t end_index = start_index + num_cols;

        for (size_t idx = start_index; idx < end_index; ++idx) {
            if (idx >= decompressed_quantized_vec.size()) throw runtime_error("[Tensor: " + tensor_name + "] Index out of bounds accessing quantized vec during reconstruction.");
            uint8_t q_wt = decompressed_quantized_vec[idx];

            size_t& iter1 = stream1_iter[group_idx][q_wt];
            size_t& iter2 = stream2_iter[group_idx][q_wt];

            // Check bounds before accessing decompressed data streams
            // Note: If a stream had TYPE_ERROR, its vector might be empty.
            const auto& stream1_data = decompressed_stream1[group_idx][q_wt];
            const auto& stream2_data = decompressed_stream2[group_idx][q_wt];

            if (iter1 >= stream1_data.size()) { throw runtime_error("Error: Read past end of decompressed stream 1 during reconstruction. Tensor: " + tensor_name + " GroupIdx: " + to_string(group_idx) + " qVal: " + to_string(q_wt) + " Index: " + to_string(idx)); }
            if (iter2 >= stream2_data.size()) { throw runtime_error("Error: Read past end of decompressed stream 2 during reconstruction. Tensor: " + tensor_name + " GroupIdx: " + to_string(group_idx) + " qVal: " + to_string(q_wt) + " Index: " + to_string(idx)); }

            uint8_t byte1 = stream1_data[iter1];
            uint8_t byte2 = stream2_data[iter2];
            output_orig_vec[idx] = (static_cast<uint16_t>(byte1) << 8) | static_cast<uint16_t>(byte2);

            iter1++;
            iter2++;
        }
    } // End loop over rows

    // Final check: ensure all stream iterators reached the end of their respective vectors
    for(uint32_t g=0; g<num_groups; ++g) {
        for(int q=0; q<256; ++q) {
            if (stream1_iter[g][q] != decompressed_stream1[g][q].size()) {
                 #pragma omp critical (cerr_orig_decomp) // Use critical if this function could be called in parallel
                 cerr << "Warning: Stream 1 iterator mismatch for tensor " << tensor_name << " Group " << g << " qVal " << q << ". Expected " << decompressed_stream1[g][q].size() << " got " << stream1_iter[g][q] << endl;
            }
             if (stream2_iter[g][q] != decompressed_stream2[g][q].size()) {
                 #pragma omp critical (cerr_orig_decomp)
                 cerr << "Warning: Stream 2 iterator mismatch for tensor " << tensor_name << " Group " << g << " qVal " << q << ". Expected " << decompressed_stream2[g][q].size() << " got " << stream2_iter[g][q] << endl;
            }
        }
    }


    return output_orig_vec;
}