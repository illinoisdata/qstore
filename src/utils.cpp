#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

#include "qstore.h"

using namespace std;

void remove_directory_if_exists(const std::string& dir_path) {
	std::filesystem::path path(dir_path);
	if (std::filesystem::exists(path)) {
		std::filesystem::remove_all(path);
		cout << "  Removed existing: " << dir_path << endl;
	}
}

void create_directory_if_not_exists(const std::string& dir_path) {
    std::filesystem::path path(dir_path);
    if (!std::filesystem::exists(path)) {
        std::cout << "Creating directory: " << dir_path << std::endl;
        std::filesystem::create_directories(path); // Creates all parent directories as needed
    }
}

size_t compute_directory_size(const std::string& dir_path) {
    size_t total_size = 0;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                total_size += std::filesystem::file_size(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error computing directory size: " << e.what() << std::endl;
    }
    return total_size;
}

string rstrip(string str, const string &chars) {
	size_t end = str.find_last_not_of(chars);
	return (end == string::npos) ? "" : str.substr(0, end + 1);
}

bool endsWith(const string &str, const string &suffix) {
	if (str.length() < suffix.length())
		return false;
	return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

void save_tensor_names(vector<string> tensor_names, string output_dir) {
    string metadata_path = output_dir + "tensor_names.txt";
    ofstream metadata_file(metadata_path);

    if (!metadata_file.is_open()) {
        cerr << "Error: Failed to open file for writing tensor names: " << metadata_path << endl;
        exit(1);
    }

    for (const auto& name : tensor_names) {
        metadata_file << name << endl;
    }
    metadata_file.close();
    cout << "Saved " << tensor_names.size() << " tensor names to " << metadata_path << endl;
}

void load_tensor_names(string tensor_names_path, vector<string>& tensor_names) {
	ifstream tensor_names_file(tensor_names_path);
    if (!tensor_names_file.is_open()) {
        cerr << "Error: Could not open tensor names file: " << tensor_names_path << endl;
        exit(1);
    }

    string name;
    while (getline(tensor_names_file, name)) {
        if (!name.empty()) {
            tensor_names.push_back(name);
        }
    }
    tensor_names_file.close();

    if (tensor_names.empty()) {
        cerr << "Error: No tensor names found in file. Exiting." << endl;
        exit(1);
    }
}