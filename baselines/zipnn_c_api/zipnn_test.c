
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "zipnn.h"

int main() {
    // Create sample data
    printf("STARTING TESTING..........\n");
    size_t num_elements = 5000*1024;
    uint16_t* data = malloc(num_elements * sizeof(uint16_t));
    
    // Initialize with some values
    for (size_t i = 0; i < num_elements; i++) {
        // data[i] = (uint16_t)(i % 65535);
        data[i] = (i % 2) ? 0 : 65535;
    }
  
    
    // Compress
    uint8_t* compressed_data = NULL;
    size_t compressed_size = 0;
    
    int compress_status = compress_uint16_data(data, 
                            num_elements, 
                                    &compressed_data, 
                                    &compressed_size, 
                                    1, 
                                    0
                                );
    if (!compress_status) {
        printf("Compression failed!\n");
        free(data);
        return 0;
    }
  
    printf("Compression successful!\n");
    printf("Original size: %zu bytes\n", num_elements * sizeof(uint16_t));
    printf("Compressed size: %zu bytes\n", compressed_size);
  
    uint8_t* decompressed_data = NULL;
    size_t decompressed_size = 0;
    int decompress_status = decompress_uint16_data(compressed_data, 
                                                  &decompressed_data, 
                                                  &decompressed_size,
                                                  1,
                                                0);
  
    if (!decompress_status) {
        printf("Decompression failed!\n");
        free(compressed_data);
        free(data);
        return 0;
    }
  
    printf("Decompression successful!\n");
    printf("Decompressed size: %zu bytes\n", decompressed_size);
    printf("Compression ratio: %.4f\n", (float)decompressed_size / compressed_size);
    
    // Verify the decompressed data
    int match = 1;
    uint16_t* decompressed_uint16 = (uint16_t*)decompressed_data;
    for (size_t i = 0; i < num_elements && match; i++) {
        if (data[i] != decompressed_uint16[i]) {
            match = 0;
            printf("Verification failed at element %zu: original=%u, decompressed=%u\n", 
                    i, data[i], decompressed_uint16[i]);
        }
    }
    
    if (match) {
        printf("Verification successful! Decompressed data matches original.\n");
    }
    
    free(decompressed_data);
    free(compressed_data);
    free(data);
    return 0;
}