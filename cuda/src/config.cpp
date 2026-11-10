#include "mppi/config.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace mppi {
namespace {   // anonymous namespace: these helpers are private to this file (like `static` in C)

std::runtime_error bad(const std::string& msg) { return std::runtime_error("config: " + msg); }

// node["key"] that throws with the key name instead of a cryptic yaml-cpp error
YAML::Node at(const YAML::Node& node, const std::string& key) {
    YAML::Node child = node[key];
    if (!child) throw bad("missing key '" + key + "'");
    return child;
}

float f(const YAML::Node& node, const std::string& key) { return at(node, key).as<float>(); }

// "vehicle.tire_model=tanh" -> raw["vehicle"]["tire_model"] = "tanh"
// The key must already exist: an override with a typo fails instead of adding
// a new key that nobody reads (Q-B1).
void apply_override(YAML::Node& raw, const std::string& kv) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) throw bad("override '" + kv + "' needs key=value");
    std::string path = kv.substr(0, eq);
    YAML::Node node = raw;
    for (auto dot = path.find('.'); dot != std::string::npos; dot = path.find('.')) {
        node.reset(at(node, path.substr(0, dot)));   // reset rebinds; '=' would copy into node
        path = path.substr(dot + 1);
    }
    at(node, path);
    node[path] = YAML::Load(kv.substr(eq + 1));
}

}  // namespace

Config load_config(const std::string& path, const std::vector<std::string>& overrides) {
    YAML::Node raw = YAML::LoadFile(path);
    for (const std::string& kv : overrides) apply_override(raw, kv);
    Config cfg{};
    Params& p = cfg.p;

    // --- model ---
    const std::string model = at(raw, "model").as<std::string>();
    if (model == "kinematic") p.model = ModelKind::Kinematic;
    else if (model == "dynamic") p.model = ModelKind::Dynamic;
    else throw bad("model='" + model + "', expected kinematic | dynamic");
    p.state_dim = at(raw, "state_dim").as<int>();
    if (p.state_dim != (p.model == ModelKind::Dynamic ? 6 : 4))
        throw bad("state_dim does not match model '" + model + "'");
    if (at(raw, "control_dim").as<int>() != CONTROL_DIM) throw bad("control_dim must be 2");

    // --- mppi ---
    const YAML::Node m = at(raw, "mppi");
    p.K = at(m, "num_samples").as<int>();
    p.T = at(m, "horizon").as<int>();
    p.dt = f(m, "dt");
    p.lambda = f(m, "lambda");
    p.gamma = f(m, "gamma");
    const YAML::Node std_ = at(m, "noise_std");
    if (!std_.IsSequence() || std_.size() != CONTROL_DIM) throw bad("noise_std needs 2 entries");
    for (int j = 0; j < CONTROL_DIM; ++j) p.noise_std[j] = std_[j].as<float>();
    cfg.seed = at(m, "seed").as<unsigned long long>();

    // --- control bounds ---
    const YAML::Node b = at(raw, "control_bounds");
    p.u_min[0] = f(b, "a_min");     p.u_max[0] = f(b, "a_max");
    p.u_min[1] = f(b, "delta_min"); p.u_max[1] = f(b, "delta_max");
    for (int j = 0; j < CONTROL_DIM; ++j)
        if (!(p.u_min[j] < p.u_max[j])) throw bad("control_bounds: min must be < max");

    // --- vehicle ---
    const YAML::Node v = at(raw, "vehicle");
    VehicleParams& veh = p.vehicle;
    veh.wheelbase = f(v, "wheelbase"); veh.lf = f(v, "lf"); veh.lr = f(v, "lr");
    veh.width = f(v, "width"); veh.mu = f(v, "mu"); veh.mass = f(v, "mass"); veh.izz = f(v, "izz");
    veh.cs_front = f(v, "cornering_stiffness_front");
    veh.cs_rear = f(v, "cornering_stiffness_rear");
    const std::string tire = at(v, "tire_model").as<std::string>();
    if (tire == "linear") veh.tire_model = TireModel::Linear;
    else if (tire == "tanh") veh.tire_model = TireModel::Tanh;
    else if (tire == "pacejka") veh.tire_model = TireModel::Pacejka;
    else throw bad("tire_model='" + tire + "', expected linear | tanh | pacejka");
    veh.pacejka_c = f(v, "pacejka_c"); veh.pacejka_e = f(v, "pacejka_e");
    veh.blend_speed_low = f(v, "blend_speed_low");
    veh.blend_speed_high = f(v, "blend_speed_high");
    if (!(0.0f < veh.blend_speed_low && veh.blend_speed_low < veh.blend_speed_high))
        throw bad("need 0 < blend_speed_low < blend_speed_high");
    const std::string integ = at(v, "integrator").as<std::string>();
    if (integ == "euler") veh.integrator = Integrator::Euler;
    else if (integ == "rk4") veh.integrator = Integrator::Rk4;
    else throw bad("integrator='" + integ + "', expected euler | rk4");
    veh.substeps = at(v, "substeps").as<int>();
    if (veh.substeps < 1) throw bad("substeps must be >= 1");
    if (std::fabs(veh.lf + veh.lr - veh.wheelbase) > 1e-6f) throw bad("lf + lr != wheelbase");

    // --- cost ---
    const YAML::Node c = at(raw, "cost");
    CostParams& cost = p.cost;
    cost.w_lateral = f(c, "w_lateral"); cost.w_progress = f(c, "w_progress");
    cost.w_offtrack = f(c, "w_offtrack"); cost.w_control = f(c, "w_control");
    cost.w_adhesion = f(c, "w_adhesion"); cost.w_speed = f(c, "w_speed");
    cost.lateral_scale = f(c, "lateral_scale"); cost.speed_scale = f(c, "speed_scale");
    cost.v_ref = f(c, "v_ref"); cost.track_half_width = f(c, "track_half_width");

    // --- paths and simulation ---
    namespace fs = std::filesystem;
    const fs::path root = fs::absolute(path).parent_path().parent_path();   // config/ -> repo
    cfg.track_bin_path = (root / at(at(raw, "track"), "bin_path").as<std::string>()).string();
    cfg.max_steps = at(at(raw, "sim"), "max_steps").as<int>();
    return cfg;
}

const char* to_string(ModelKind m) { return m == ModelKind::Dynamic ? "dynamic" : "kinematic"; }

const char* to_string(TireModel t) {
    switch (t) {
        case TireModel::Linear: return "linear";
        case TireModel::Tanh: return "tanh";
        case TireModel::Pacejka: return "pacejka";
    }
    return "?";
}

const char* to_string(Integrator i) { return i == Integrator::Rk4 ? "rk4" : "euler"; }

}  // namespace mppi
