#include "mppi/track.cuh"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mppi {

// Header written by track.save: struct "<4sIiiffff", 32 bytes, little-endian.
TrackHost load_track(const std::string& bin_path) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) throw std::runtime_error("track: cannot open " + bin_path + " (run `pixi run track`)");

    char magic[4];
    std::uint32_t version = 0;
    TrackHost tr;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&tr.nx), 4);
    in.read(reinterpret_cast<char*>(&tr.ny), 4);
    in.read(reinterpret_cast<char*>(&tr.x_min), 4);
    in.read(reinterpret_cast<char*>(&tr.y_min), 4);
    in.read(reinterpret_cast<char*>(&tr.res), 4);
    in.read(reinterpret_cast<char*>(&tr.length), 4);
    if (!in || std::memcmp(magic, "MPPG", 4) != 0 || version != 1)
        throw std::runtime_error("track: bad header in " + bin_path);
    if (tr.nx <= 0 || tr.ny <= 0 || !(tr.res > 0.0f))
        throw std::runtime_error("track: bad grid geometry in " + bin_path);

    tr.grid.resize(static_cast<std::size_t>(tr.nx) * tr.ny * 2);
    in.read(reinterpret_cast<char*>(tr.grid.data()), tr.grid.size() * sizeof(float));
    if (!in) throw std::runtime_error("track: " + bin_path + " is truncated");
    return tr;
}

}  // namespace mppi
