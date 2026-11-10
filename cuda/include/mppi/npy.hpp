// Minimal .npy reader/writer: little-endian float32 or float64, C order.
//
// Enough to exchange arrays with NumPy (np.save / np.load) without a library.
// Format: magic "\x93NUMPY", version 1.0, a 2-byte header length, then a
// Python dict literal such as {'descr': '<f4', 'fortran_order': False,
// 'shape': (1024, 30, 2), }, padded with spaces to a multiple of 64 bytes.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mppi::npy {

template <class T>
const char* descr() {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "float or double only");
    return std::is_same_v<T, float> ? "<f4" : "<f8";
}

template <class T>
struct Array {
    std::vector<std::size_t> shape;
    std::vector<T> data;
};

template <class T>
Array<T> load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("npy: cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0 || magic[6] != 1)
        throw std::runtime_error("npy: " + path + " is not a version 1.0 .npy file");
    std::uint16_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 2);
    std::string header(len, ' ');
    in.read(header.data(), len);

    if (header.find(std::string("'descr': '") + descr<T>() + "'") == std::string::npos)
        throw std::runtime_error("npy: " + path + " has dtype " + header.substr(0, 40) +
                                 ", expected " + descr<T>());
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("npy: " + path + " is Fortran ordered");

    Array<T> arr;
    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    std::string dims = header.substr(open + 1, close - open - 1);   // "1024, 30, 2" or "6,"
    std::size_t count = 1, pos = 0;
    while (pos < dims.size()) {
        const auto comma = dims.find(',', pos);
        const std::string tok = dims.substr(pos, comma - pos);
        if (tok.find_first_not_of(' ') != std::string::npos) {
            arr.shape.push_back(std::stoul(tok));
            count *= arr.shape.back();
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    arr.data.resize(count);
    in.read(reinterpret_cast<char*>(arr.data.data()), count * sizeof(T));
    if (!in) throw std::runtime_error("npy: " + path + " is truncated");
    return arr;
}

template <class T>
void save(const std::string& path, const T* data, const std::vector<std::size_t>& shape) {
    std::string dims;
    std::size_t count = 1;
    for (std::size_t d : shape) { dims += std::to_string(d) + ", "; count *= d; }
    if (shape.size() > 1) dims.resize(dims.size() - 1);   // "(3, 4)" but "(6,)"
    else if (shape.empty()) dims.clear();
    std::string header = std::string("{'descr': '") + descr<T>() +
                         "', 'fortran_order': False, 'shape': (" + dims + "), }";
    header.append(64 - (10 + header.size() + 1) % 64, ' ');
    header += '\n';
    const std::uint16_t len = static_cast<std::uint16_t>(header.size());

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("npy: cannot write " + path);
    out.write("\x93NUMPY\x01\x00", 8);
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(data), count * sizeof(T));
}

template <class T>
void save(const std::string& path, const std::vector<T>& data, const std::vector<std::size_t>& shape) {
    save(path, data.data(), shape);
}

}  // namespace mppi::npy
