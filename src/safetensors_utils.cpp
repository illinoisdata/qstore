//
// Created by mfuntowicz on 3/28/23.
//
// ! Requires std=c++20
#include "safetensors.h"

namespace huggingface::safetensors {

	std::string dtype_to_string(huggingface::safetensors::dtype dtype) {
		switch (dtype) {
		case huggingface::safetensors::kBOOL:
			return "BOOL";
		case huggingface::safetensors::kUINT_8:
			return "U8";
		case huggingface::safetensors::kINT_8:
			return "I8";
		case huggingface::safetensors::kINT_16:
			return "I16";
		case huggingface::safetensors::kUINT_16:
			return "U16";
		case huggingface::safetensors::kFLOAT_16:
			return "F16";
		case huggingface::safetensors::kBFLOAT_16:
			return "BF16";
		case huggingface::safetensors::kINT_32:
			return "I32";
		case huggingface::safetensors::kUINT_32:
			return "U32";
		case huggingface::safetensors::kFLOAT_32:
			return "F32";
		case huggingface::safetensors::kFLOAT_64:
			return "F64";
		case huggingface::safetensors::kINT_64:
			return "I64";
		case huggingface::safetensors::kUINT_64:
			return "U64";
		default:
			return "Unknown"; // In case of an invalid enum value
		}
	}

	safetensors_t deserialize(std::basic_istream<char> &in) {
		uint64_t header_size = 0;

		// todo: handle exception
		in.read(reinterpret_cast<char *>(&header_size), sizeof header_size);

		std::vector<char> meta_block(header_size);
		in.read(meta_block.data(), static_cast<std::streamsize>(header_size));
		const auto metadatas = json::parse(meta_block);

		// How many bytes remaining to pre-allocate the storage tensor
		in.seekg(0, std::ios::end);
		std::streamsize f_size = in.tellg();
		in.seekg(8 + header_size, std::ios::beg);
		const auto tensors_size = f_size - 8 - header_size;

		auto metas_table = std::unordered_map<std::string, const metadata_t>(metadatas.size());
		auto tensors_storage = std::vector<char>(tensors_size);

		// Read the remaining content
		in.read(tensors_storage.data(), static_cast<std::streamsize>(tensors_size));

		// Populate the meta lookup table
		if (metadatas.is_object()) {
			for (auto &item : metadatas.items()) {
				if (item.key() != "__metadata__") {
					const auto name = std::string(item.key());
					const auto &info = item.value();

					const metadata_t meta = {info["dtype"].get<dtype>(), info["shape"],
											info["data_offsets"]};
					metas_table.insert(std::pair(name, meta));
				}
			}
		}

		return {metas_table, tensors_storage};
	}

	safetensors_t::safetensors_t(std::unordered_map<std::string, const metadata_t> &metas,
								std::vector<char> &storage)
		: metas(metas), storage(storage) {}

	span<const char> safetensors_t::operator[](const char *name) const {
		const auto meta = metas.at(name);
		const auto [t_begin, t_end] = meta.data_offsets;
		return {storage.begin() + static_cast<ptrdiff_t>(t_begin),
				storage.begin() + static_cast<ptrdiff_t>(t_end)};
	}
} // namespace huggingface::safetensors