#ifndef QSTORE_H
#define QSTORE_H

#include <string>
#include <vector>

#include <nonstd/span.hpp>

using nonstd::span;
using namespace std;

constexpr int NUM_THREADS = 48;
constexpr int CHUNK_SIZE = 1024 * 100;

void remove_directory_if_exists(const std::string& dir_path);
void create_directory_if_not_exists(const std::string& dir_path);
size_t compute_directory_size(const std::string& dir_path);
void save_tensor_names(vector<string> tensor_names, string output_dir);
void load_tensor_names(string tensor_names_path, vector<string>& tensor_names);
std::string rstrip(std::string str, const std::string &chars);
bool endsWith(const std::string &str, const std::string &suffix);

#endif // QSTORE_H