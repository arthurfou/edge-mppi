// Checkpoint B: load the config and the track, print what Python prints.
//
//   ./build/check_config [config/mppi.yaml] [section.key=value ...]
//
// Expected with the default config: length 48.31 m, grid 454x279.
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "mppi/config.hpp"
#include "mppi/npy.hpp"
#include "mppi/track.cuh"

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "config/mppi.yaml";
    const std::vector<std::string> overrides(argv + (argc > 1 ? 2 : 1), argv + argc);

    try {
        const mppi::Config cfg = mppi::load_config(path, overrides);
        const mppi::Params& p = cfg.p;
        std::printf("model       %s (state_dim %d)\n", mppi::to_string(p.model), p.state_dim);
        std::printf("K, T        %d, %d   dt %.3f\n", p.K, p.T, p.dt);
        std::printf("lambda      %.3f   gamma %.3f\n", p.lambda, p.gamma);
        std::printf("noise_std   [%.2f, %.2f]\n", p.noise_std[0], p.noise_std[1]);
        std::printf("tire_model  %s   integrator %s x%d\n", mppi::to_string(p.vehicle.tire_model),
                    mppi::to_string(p.vehicle.integrator), p.vehicle.substeps);
        std::printf("v_ref       %.2f m/s\n", p.cost.v_ref);

        const mppi::TrackHost tr = mppi::load_track(cfg.track_bin_path);
        std::printf("track       %s\n", cfg.track_bin_path.c_str());
        std::printf("grid        %dx%d   res %.3f m   origin (%.3f, %.3f)\n", tr.nx, tr.ny, tr.res,
                    tr.x_min, tr.y_min);
        std::printf("length      %.2f m\n", tr.length);

        // A few values to compare with Python (track.lookup, track.wrap).
        const mppi::TrackView view = tr.view();
        const float pts[3][2] = {{0.0f, 0.0f}, {5.0f, 3.0f}, {-100.0f, 100.0f}};
        for (const auto& pt : pts) {
            float d, s;
            mppi::lookup(view, pt[0], pt[1], d, s);
            std::printf("lookup(%7.2f, %7.2f) -> d %.6f, s %.6f\n", pt[0], pt[1], d, s);
        }
        std::printf("wrap(-3, 10) = %g   wrap(30, %.2f) = %g\n", mppi::wrap(-3.0f, 10.0f), tr.length,
                    mppi::wrap(30.0f, tr.length));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
