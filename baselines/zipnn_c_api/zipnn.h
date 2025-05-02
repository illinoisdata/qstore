#ifndef ZIPNN_CORE_H
#define ZIPNN_CORE_H

#include <stdint.h>
#include <stddef.h>

// Error codes
#define ZIPNN_SUCCESS 0
#define ZIPNN_ERROR_MEMORY 1
#define ZIPNN_ERROR_THREAD 2
#define ZIPNN_ERROR_INVALID_PARAM 3

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compresses data using the zipnn algorithm
 */
int zipnn_compress(
    uint8_t *header, size_t header_len,
    uint8_t *data, size_t data_len,
    uint32_t numBuf,
    uint32_t bits_mode, 
    uint32_t bytes_mode, 
    uint32_t is_redata,
    size_t origChunkSize,
    float compThreshold,
    uint32_t checkThAfterPercent,
    uint32_t threads,
    uint8_t **out_buffer,
    size_t *out_buffer_size
);

/*
 * Decompresses data previously compressed with zipnn_compress
 */
int zipnn_decompress(
    uint8_t *compressed_data, size_t compressed_data_len,
    uint32_t numBuf,
    uint32_t bits_mode, 
    uint32_t bytes_mode,
    size_t origChunkSize,
    size_t origSize,
    uint32_t threads,
    uint8_t **out_buffer,
    size_t *out_buffer_size
);

int compress_uint16_data(
    uint16_t* data, 
    size_t data_len_elements,
    uint8_t** out_buffer, 
    size_t* out_buffer_size,
    uint32_t threads,
    uint32_t bits_mode
);

int decompress_uint16_data(
    uint8_t* compressed_data, 
    uint8_t** decompressed_data, 
    size_t* decompressed_size,
    uint32_t threads,
    uint32_t bits_mode
);

#ifdef __cplusplus
}
#endif
#endif /* ZIPNN_CORE_H */