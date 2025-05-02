
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

#include <nonstd/span.hpp>

#include "safetensors.h"
#include "qstore.h"

using nonstd::span;
using namespace std;

// Helper to safely cast span<const char> to span<const T>
template <typename T>
span<const T> as_typed_span(span<const char> s) {
    if (s.empty()) {
        return span<const T>();
    }
    if (s.size() % sizeof(T) != 0) {
        cerr << "Warning: Invalid span size (" << s.size()
             << ") for type casting to size " << sizeof(T) << ". Returning empty span." << endl;
        return span<const T>();
    }
    return span<const T>(reinterpret_cast<const T*>(s.data()), s.size() / sizeof(T));
}


struct MixedPrecMetadata {
    // --- File Path Maps ---
    unordered_map<string, string> orig_wt_file_map;
    unordered_map<string, string> q_wt_file_map;
    unordered_map<string, string> scale_file_map;

    vector<string> tensor_names;

    // --- Final Data Storage (Vectors) ---
    // Maps from tensor name to a vector containing a copy of the data.
    unordered_map<string, vector<uint8_t>> q_data_vectors;
    unordered_map<string, vector<uint16_t>> orig_data_vectors;
    unordered_map<string, vector<uint32_t>> scale_data_vectors;
    unordered_map<string, vector<size_t>> tensor_shapes; // Shapes are small, vector is fine

    // --- Internal/Temporary Storage during loading ---
    // These hold the actual data buffers read from files.
    // They are needed while spans/vectors are created but could potentially
    // be cleared after vectors are populated if memory is extremely tight,
    // though keeping them might be useful for debugging or other operations.
    unordered_map<string, huggingface::safetensors::safetensors_t> q_safetensors_map;
    unordered_map<string, huggingface::safetensors::safetensors_t> orig_safetensors_map;
    unordered_map<string, huggingface::safetensors::safetensors_t> scale_safetensors_map;
};

MixedPrecMetadata load_tensors_and_metadata(string orig_model_dir, string quantized_model_dir);
