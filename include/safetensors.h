//
// Created by mfuntowicz on 3/28/23.
//

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

// #include <span>
#include "nlohmann/json.hpp"
#include <nonstd/span.hpp>

using nonstd::span;
using json = nlohmann::json;

namespace huggingface::safetensors {
enum dtype {
	/// Boolean type
	kBOOL,
	/// Unsigned byte
	kUINT_8,
	/// Signed byte
	kINT_8,
	/// Signed integer (16-bit)
	kINT_16,
	/// Unsigned integer (16-bit)
	kUINT_16,
	/// Half-precision floating point
	kFLOAT_16,
	/// Brain floating point
	kBFLOAT_16,
	/// Signed integer (32-bit)
	kINT_32,
	/// Unsigned integer (32-bit)
	kUINT_32,
	/// Floating point (32-bit)
	kFLOAT_32,
	/// Floating point (64-bit)
	kFLOAT_64,
	/// Signed integer (64-bit)
	kINT_64,
	/// Unsigned integer (64-bit)
	kUINT_64,
};

NLOHMANN_JSON_SERIALIZE_ENUM(dtype, {
                                        {kBOOL, "BOOL"},
                                        {kUINT_8, "U8"},
                                        {kINT_8, "I8"},
                                        {kINT_16, "I16"},
                                        {kUINT_16, "U16"},
                                        {kFLOAT_16, "F16"},
                                        {kBFLOAT_16, "BF16"},
                                        {kINT_32, "I32"},
                                        {kUINT_32, "U32"},
                                        {kFLOAT_32, "F32"},
                                        {kFLOAT_64, "F64"},
                                        {kINT_64, "I64"},
                                        {kUINT_64, "U64"},
                                    })

struct metadata_t {
	dtype data_type;
	std::vector<size_t> shape;
	std::pair<size_t, size_t> data_offsets;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metadata_t, data_type, shape, data_offsets)

/**
 *
 */
class safetensors_t {
  private:
	const std::unordered_map<std::string, const metadata_t> metas;
	const std::vector<char> storage;

  public:
	safetensors_t(std::unordered_map<std::string, const metadata_t> &, std::vector<char> &);

	const std::unordered_map<std::string, const metadata_t> &get_metas() const { return metas; }
	/**
	 *
	 * @return
	 */
	inline size_t size() const { return metas.size(); }

	/**
	 *
	 * @param name
	 * @return
	 */
	span<const char> operator[](const char *name) const;
};

/**
 *
 * @param in
 * @return
 */
safetensors_t deserialize(std::basic_istream<char> &in);

std::string dtype_to_string(huggingface::safetensors::dtype dtype);
} // namespace huggingface::safetensors

#endif // SAFETENSORS_H
