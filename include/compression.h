#include <vector>
#include <string>
#include <cstdint>
#include <fstream>

// Define the max source size for HUF_compress
// Check FiniteStateEntropy/lib/huf_compress.c (128*1024)
#ifndef HUF_COMPRESS_SRCSIZE_MAX
#define HUF_COMPRESS_SRCSIZE_MAX (128 * 1024)
#endif

#define QUANTIZED_CHUNK_SIZE (128 * 1024) // Chunk size for quantized

// Constants for compression type
#define TYPE_UNCOMPRESSED 0
#define TYPE_COMPRESSED_HUF 1
#define TYPE_CHUNKED_HUF 2    // Type for chunked original streams
#define TYPE_ERROR 255

// Structure to hold results for individual data chunks
struct ChunkResult {
    uint8_t* data = nullptr;          // Pointer to compressed (or uncompressed) data (malloc'd)
    size_t compressed_size = 0;       // Size of the data buffer
    size_t original_size = 0;         // Original size of this chunk
    uint8_t type = TYPE_ERROR;        // 0=uncompressed, 1=HUF compressed, 2=chunked container, 255=error
};

// Structure to hold results read from file (for decompression pipeline)
struct CompressedDataBuffer {
    std::string tensor_name;
    std::vector<uint8_t> buffer; // Holds the entire file content OR a segment
    bool success = false;
    std::string error_msg;
};

// --- Function Declarations ---

// Quantized Tensor Compression/Serialization/Decompression
void compress_quantized_tensor_chunked(
    const std::vector<uint8_t>& q_data_vec,
    std::vector<ChunkResult>& chunk_results,
    size_t chunk_size = QUANTIZED_CHUNK_SIZE
);

// Writes the chunked quantized tensor data to the stream and returns bytes written.
// Returns 0 on error.
uint64_t serialize_quantized_tensor(
    std::ofstream& outFile,
    const std::string& tensor_name, // For error messages
    const std::vector<ChunkResult>& chunk_results,
    const std::vector<size_t>& tensor_shape
);

// Reads a specific segment from an open file stream into a buffer.
CompressedDataBuffer read_segment_to_buffer(
    std::ifstream& inFile, // Pass open file stream
    const std::string& name,
    uint64_t offset,
    uint64_t size
);

std::vector<uint8_t> decompress_quantized_tensor_chunked(
    const std::vector<uint8_t>& compressed_buffer_vec, // Buffer for ONE tensor segment
    const std::string& tensor_name
);

// Original Tensor Compression/Serialization/Decompression
void compress_tensor_optimized(
    const std::vector<uint16_t>& input_orig_vec,
    const std::vector<uint8_t>& input_quantized_vec,
    const std::vector<uint32_t>& scale_data_vec,
    std::vector<std::vector<uint8_t*>>& result_data_ptr1,
    std::vector<std::vector<size_t>>& result_metadata1,
    std::vector<std::vector<uint8_t*>>& result_data_ptr2,
    std::vector<std::vector<size_t>>& result_metadata2,
    std::vector<std::vector<uint8_t>>& result_type1,
    std::vector<std::vector<uint8_t>>& result_type2,
    std::vector<std::vector<size_t>>& result_original_size1,
    std::vector<std::vector<size_t>>& result_original_size2,
    std::vector<uint32_t>& row_to_group_idx_map_out,
    const std::vector<size_t>& tensor_shape,
    size_t& total_orig_bytes_out,
    size_t& total_comp_bytes_out
);

// Writes the grouped/chunked original tensor data to the stream and returns bytes written.
// Returns 0 on error.
uint64_t serialize_original_tensor(
    std::ofstream& outFile,
    const std::string& tensor_name, // For error messages
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
);

std::vector<uint16_t> decompress_tensor_from_buffer(
    const std::vector<uint8_t>& compressed_buffer_vec, // Buffer for ONE segment
    const std::string& tensor_name,
    const std::vector<uint8_t>& decompressed_quantized_vec // Auxiliary data
);