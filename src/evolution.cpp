// evolution.cpp - Core evolution algorithm for lattice fields

#include "main.h"
#include <algorithm>
#include <limits>

// -------------------- Laplacians --------------------

// Helper for periodic indexing
inline int INCREMENT(int i) {
  return (i == N - 1) ? 0 : i + 1;
}

// Decrement index with periodic wrapping (i → i-1 mod N)
inline int DECREMENT(int i) {
  return (i == 0) ? N - 1 : i - 1;
}

// Generic 7-point Laplacian for real-space arrays 
template <typename T>
inline T lapl(int i, int j, int k, const std::vector<T>& field, int n) {
  if (i == 0 || j == 0 || k == 0 || i == N - 1 || j == N - 1 || k == N - 1) {
    return (
      field[idx_mf(n,i,j,INCREMENT(k))] + field[idx_mf(n,i,j,DECREMENT(k))] +
      field[idx_mf(n,i,INCREMENT(j),k)] + field[idx_mf(n,i,DECREMENT(j),k)] +
      field[idx_mf(n,INCREMENT(i),j,k)] + field[idx_mf(n,DECREMENT(i),j,k)] -
      T(6) * field[idx_mf(n,i,j,k)]
    );
  } else {
    return (
      field[idx_mf(n,i,j,k+1)] + field[idx_mf(n,i,j,k-1)] +
      field[idx_mf(n,i,j+1,k)] + field[idx_mf(n,i,j-1,k)] +
      field[idx_mf(n,i+1,j,k)] + field[idx_mf(n,i-1,j,k)] -
      T(6) * field[idx_mf(n,i,j,k)]
    );
  }
}


// -------------------- Energy Calculations --------------------

// Compute gradient energy density (averaged)
float gradient_energy(const std::vector<double>& field, double a) {
  DECLARE_INDICES
  float gradient = 0.;
  float norm = pw2(1. / (a * dx));
  LOOP_MF {
    double partial = 0.0;
#if parallel_calculation
#pragma omp parallel for reduction(+:partial) collapse(3)
#endif
    LOOP { partial += field[idx_mf(n,i,j,k)] * lapl(i, j, k, field, n); }
    gradient -= (float)partial;
  }
  return 0.5 * gradient * norm / (float)gridsize;
}

// Compute kinetic energy density (averaged)
double kin_energy(
    const std::vector<double>& velocity,
    double scale_factor
) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long id = 0; id < static_cast<long long>(velocity.size()); ++id) {
        sum += velocity[id] * velocity[id];
    }

    return 0.5
        * std::pow(scale_factor, 2.0 * rescale_s - 2.0)
        * sum / static_cast<double>(gridsize);
}

double scale_acceleration(
    const std::vector<double>& field,
    const std::vector<double>& velocity,
    double scale_factor,
    double scale_derivative
) {
    const double kinetic = kin_energy(velocity, scale_factor);
    const double gradient = gradient_energy(field, scale_factor);
    const double potential_value = potential_energy(field);

    const double rho_minus_3p =
        -2.0 * kinetic
        + 2.0 * gradient
        + 4.0 * potential_value;

    return
        -rescale_s * scale_derivative * scale_derivative / scale_factor
        + std::pow(scale_factor, 3.0 - 2.0 * rescale_s)
          * rho_minus_3p / 6.0;
}

// -------------------- Main Field Evolution --------------------

// Evaluate the coupled lattice and scale-factor equations for one RK45 stage.
void evaluate_lattice_rhs(
  const std::vector<double>& f,
  const std::vector<double>& fd,
  double a,
  double ad,
  std::vector<double>& k_f,
  std::vector<double>& k_fd,
  double& k_a,
  double& k_ad
)
{
  DECLARE_INDICES
  #pragma omp parallel for collapse(4) private(n,i,j,k)
  for (n = 0; n < nFields; n++)
    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
        for (k = 0; k < N; k++) {
      k_f[idx_mf(n,i,j,k)] = fd[idx_mf(n,i,j,k)];
      k_fd[idx_mf(n,i,j,k)] = pow(a, -2.0*rescale_s)/pw2(dx) * lapl(i, j, k, f, n)
                                  - (2.0+rescale_s)*ad*fd[idx_mf(n,i,j,k)]/a
                                  - pow(a, 2.0-2.0*rescale_s) * potential_derivative(f[idx_mf(n,i,j,k)],n,i,j,k, &f);
    }

  k_a = ad;
  k_ad = scale_acceleration(f, fd, a, ad);
  
}

// -------------------- RK45 Integrator --------------------

struct RK45StepResult {
  bool accepted;
  double error_norm;
  double suggested_step;
};

struct RK45Workspace {
  std::vector<double> k1_f, k2_f, k3_f, k4_f, k5_f, k6_f;
  std::vector<double> k1_fd, k2_fd, k3_fd, k4_fd, k5_fd, k6_fd;
  std::vector<double> f_s, fd_s;

  explicit RK45Workspace(size_t size)
      : k1_f(size), k2_f(size), k3_f(size), k4_f(size), k5_f(size), k6_f(size),
        k1_fd(size), k2_fd(size), k3_fd(size), k4_fd(size), k5_fd(size), k6_fd(size),
        f_s(size), fd_s(size) {}
};

// Rejected attempts leave every global dynamical variable unchanged
static RK45StepResult rk45_evolve(double d, RK45Workspace& workspace) {
  auto& k1_f = workspace.k1_f;
  auto& k2_f = workspace.k2_f;
  auto& k3_f = workspace.k3_f;
  auto& k4_f = workspace.k4_f;
  auto& k5_f = workspace.k5_f;
  auto& k6_f = workspace.k6_f;
  auto& k1_fd = workspace.k1_fd;
  auto& k2_fd = workspace.k2_fd;
  auto& k3_fd = workspace.k3_fd;
  auto& k4_fd = workspace.k4_fd;
  auto& k5_fd = workspace.k5_fd;
  auto& k6_fd = workspace.k6_fd;
  auto& f_s = workspace.f_s;
  auto& fd_s = workspace.fd_s;

  double k1_a,  k2_a,  k3_a,  k4_a,  k5_a,  k6_a;
  double k1_ad, k2_ad, k3_ad, k4_ad, k5_ad, k6_ad;
  double a_s, ad_s, a5_, ad5_;

  // Stage 1: k1 at current state
  evaluate_lattice_rhs(f, fd, a, ad, k1_f, k1_fd, k1_a, k1_ad);

  // Stage 2: c2 = 1/5
#if parallel_calculation
#pragma omp parallel for
#endif
  for (long long id = 0; id < static_cast<long long>(f.size()); ++id) {
    f_s[id] = f[id] + d * (1.0/5.0 * k1_f[id]);
    fd_s[id] = fd[id] + d * (1.0/5.0 * k1_fd[id]);
  }
  a_s  = a  + d * (1.0/5.0 * k1_a);
  ad_s = ad + d * (1.0/5.0 * k1_ad);
  evaluate_lattice_rhs(f_s, fd_s, a_s, ad_s, k2_f, k2_fd, k2_a, k2_ad);

  // Stage 3: c3 = 3/10
#if parallel_calculation
#pragma omp parallel for
#endif
  for (long long id = 0; id < static_cast<long long>(f.size()); ++id) {
    f_s[id] = f[id] + d * (3.0/40.0 * k1_f[id] + 9.0/40.0 * k2_f[id]);
    fd_s[id] = fd[id] + d * (3.0/40.0 * k1_fd[id] + 9.0/40.0 * k2_fd[id]);
  }
  a_s  = a  + d * (3.0/40.0 * k1_a  + 9.0/40.0 * k2_a);
  ad_s = ad + d * (3.0/40.0 * k1_ad + 9.0/40.0 * k2_ad);
  evaluate_lattice_rhs(f_s, fd_s, a_s, ad_s, k3_f, k3_fd, k3_a, k3_ad);

  // Stage 4: c4 = 4/5
#if parallel_calculation
#pragma omp parallel for
#endif
  for (long long id = 0; id < static_cast<long long>(f.size()); ++id) {
    f_s[id] = f[id] + d * (44.0/45.0 * k1_f[id] - 56.0/15.0 * k2_f[id] + 32.0/9.0 * k3_f[id]);
    fd_s[id] = fd[id] + d * (44.0/45.0 * k1_fd[id] - 56.0/15.0 * k2_fd[id] + 32.0/9.0 * k3_fd[id]);
  }
  a_s  = a  + d * (44.0/45.0 * k1_a  - 56.0/15.0 * k2_a  + 32.0/9.0 * k3_a);
  ad_s = ad + d * (44.0/45.0 * k1_ad - 56.0/15.0 * k2_ad + 32.0/9.0 * k3_ad);
  evaluate_lattice_rhs(f_s, fd_s, a_s, ad_s, k4_f, k4_fd, k4_a, k4_ad);

  // Stage 5: c5 = 8/9
#if parallel_calculation
#pragma omp parallel for
#endif
  for (long long id = 0; id < static_cast<long long>(f.size()); ++id) {
    f_s[id] = f[id] + d * (19372.0/6561.0 * k1_f[id] - 25360.0/2187.0 * k2_f[id] + 64448.0/6561.0 * k3_f[id] - 212.0/729.0 * k4_f[id]);
    fd_s[id] = fd[id] + d * (19372.0/6561.0 * k1_fd[id] - 25360.0/2187.0 * k2_fd[id] + 64448.0/6561.0 * k3_fd[id] - 212.0/729.0 * k4_fd[id]);
  }
  a_s  = a  + d * (19372.0/6561.0 * k1_a  - 25360.0/2187.0 * k2_a  + 64448.0/6561.0 * k3_a  - 212.0/729.0 * k4_a);
  ad_s = ad + d * (19372.0/6561.0 * k1_ad - 25360.0/2187.0 * k2_ad + 64448.0/6561.0 * k3_ad - 212.0/729.0 * k4_ad);
  evaluate_lattice_rhs(f_s, fd_s, a_s, ad_s, k5_f, k5_fd, k5_a, k5_ad);

  // Stage 6: c6 = 1
#if parallel_calculation
#pragma omp parallel for
#endif
  for (long long id = 0; id < static_cast<long long>(f.size()); ++id) {
    f_s[id] = f[id] + d * (9017.0/3168.0 * k1_f[id] - 355.0/33.0 * k2_f[id] + 46732.0/5247.0 * k3_f[id] + 49.0/176.0 * k4_f[id] - 5103.0/18656.0 * k5_f[id]);
    fd_s[id] = fd[id] + d * (9017.0/3168.0 * k1_fd[id] - 355.0/33.0 * k2_fd[id] + 46732.0/5247.0 * k3_fd[id] + 49.0/176.0 * k4_fd[id] - 5103.0/18656.0 * k5_fd[id]);
  }
  a_s  = a  + d * (9017.0/3168.0 * k1_a  - 355.0/33.0 * k2_a  + 46732.0/5247.0 * k3_a  + 49.0/176.0 * k4_a  - 5103.0/18656.0 * k5_a);
  ad_s = ad + d * (9017.0/3168.0 * k1_ad - 355.0/33.0 * k2_ad + 46732.0/5247.0 * k3_ad + 49.0/176.0 * k4_ad - 5103.0/18656.0 * k5_ad);
  evaluate_lattice_rhs(f_s, fd_s, a_s, ad_s, k6_f, k6_fd, k6_a, k6_ad);

  // 5th-order solution (b5 weights)
#if parallel_calculation
#pragma omp parallel for
#endif
  for (long long id = 0; id < static_cast<long long>(f.size()); ++id) {
    f_s[id] = f[id] + d * (35.0/384.0 * k1_f[id] + 500.0/1113.0 * k3_f[id] + 125.0/192.0 * k4_f[id] - 2187.0/6784.0 * k5_f[id] + 11.0/84.0 * k6_f[id]);
    fd_s[id] = fd[id] + d * (35.0/384.0 * k1_fd[id] + 500.0/1113.0 * k3_fd[id] + 125.0/192.0 * k4_fd[id] - 2187.0/6784.0 * k5_fd[id] + 11.0/84.0 * k6_fd[id]);
  }
  a5_  = a  + d * (35.0/384.0 * k1_a  + 500.0/1113.0 * k3_a  + 125.0/192.0 * k4_a  - 2187.0/6784.0 * k5_a  + 11.0/84.0 * k6_a);
  ad5_ = ad + d * (35.0/384.0 * k1_ad + 500.0/1113.0 * k3_ad + 125.0/192.0 * k4_ad - 2187.0/6784.0 * k5_ad + 11.0/84.0 * k6_ad);

  // Stage 7: k7 at the fifth-order solution.  Reuse the now-dead k2 buffers.
  evaluate_lattice_rhs(f_s, fd_s, a5_, ad5_, k2_f, k2_fd, k2_a, k2_ad);

  // Error coefficients: e = b5 - b4 
  const double e1 =    71.0/57600.0;
  const double e3 =   -71.0/16695.0;
  const double e4 =    71.0/1920.0;
  const double e5 = -17253.0/339200.0;
  const double e6 =    22.0/525.0;
  const double e7 =    -1.0/40.0;

  double err = 0.0;
  int invalid_candidate =
      (!std::isfinite(a5_) || !std::isfinite(ad5_) || a5_ <= 0.0) ? 1 : 0;

  for (int n = 0; n < nFields; ++n) {
    double field_error_sum = 0.0;
    double velocity_error_sum = 0.0;
    int block_invalid = 0;
#if parallel_calculation
#pragma omp parallel for \
    reduction(+:field_error_sum,velocity_error_sum) reduction(max:block_invalid)
#endif
    for (long long lattice_id = 0;
         lattice_id < static_cast<long long>(gridsize); ++lattice_id) {
      const size_t id = static_cast<size_t>(n) * gridsize + lattice_id;
      const double field_error = d * (
          e1*k1_f[id] + e3*k3_f[id] + e4*k4_f[id]
          + e5*k5_f[id] + e6*k6_f[id] + e7*k2_f[id]);
      const double velocity_error = d * (
          e1*k1_fd[id] + e3*k3_fd[id] + e4*k4_fd[id]
          + e5*k5_fd[id] + e6*k6_fd[id] + e7*k2_fd[id]);
      const double field_scale = rk45_atol + rk45_rtol
          * std::max(std::abs(f[id]), std::abs(f_s[id]));
      const double velocity_scale = rk45_atol + rk45_rtol
          * std::max(std::abs(fd[id]), std::abs(fd_s[id]));
      const double field_ratio = field_error / field_scale;
      const double velocity_ratio = velocity_error / velocity_scale;

      if (!std::isfinite(f_s[id]) || !std::isfinite(fd_s[id])
          || !std::isfinite(field_ratio) || !std::isfinite(velocity_ratio)) {
        block_invalid = 1;
      } else {
        field_error_sum += field_ratio * field_ratio;
        velocity_error_sum += velocity_ratio * velocity_ratio;
      }
    }

    invalid_candidate = std::max(invalid_candidate, block_invalid);
    err = std::max(err, std::sqrt(
        field_error_sum / static_cast<double>(gridsize)));
    err = std::max(err, std::sqrt(
        velocity_error_sum / static_cast<double>(gridsize)));
  }

  const double a_error = d * (
      e1*k1_a + e3*k3_a + e4*k4_a
      + e5*k5_a + e6*k6_a + e7*k2_a);
  const double ad_error = d * (
      e1*k1_ad + e3*k3_ad + e4*k4_ad
      + e5*k5_ad + e6*k6_ad + e7*k2_ad);
  const double a_scale = rk45_geometry_atol + rk45_rtol
      * std::max(std::abs(a), std::abs(a5_));
  const double ad_scale = rk45_geometry_atol + rk45_rtol
      * std::max(std::abs(ad), std::abs(ad5_));
  const double a_ratio = a_error / a_scale;
  const double ad_ratio = ad_error / ad_scale;
  if (!std::isfinite(a_ratio) || !std::isfinite(ad_ratio)) {
    invalid_candidate = 1;
  } else {
    err = std::max(err, std::abs(a_ratio));
    err = std::max(err, std::abs(ad_ratio));
  }
  if (invalid_candidate) {
    err = std::numeric_limits<double>::infinity();
  }

  const bool accepted = err <= 1.0;
  double factor = rk45_min_factor;
  if (std::isfinite(err)) {
    factor = (err == 0.0)
        ? rk45_max_factor
        : rk45_safety * std::pow(err, -0.2);
  }
  const double maximum_factor = accepted ? rk45_max_factor : 1.0;
  factor = std::clamp(factor, rk45_min_factor, maximum_factor);
  const double d_new = d * factor;

  if (!accepted) {
    return {false, err, d_new};
  }

  f.swap(f_s);
  fd.swap(fd_s);
  a = a5_;
  ad = ad5_;
  ad2 = k2_ad; // k7 is the RHS evaluated at this accepted state
  t += d;

  return {true, err, d_new};
}

// -------------------- Main Evolution Loop --------------------

static double rk45_step_from_base(double base_step, double scale_factor) {
  return base_step * std::pow(scale_factor, rescale_s - 1.0);
}

static double rk45_minimum_step(double scale_factor) {
  return rk45_step_from_base(rk45_dt_min, scale_factor);
}

static double rk45_maximum_step(double scale_factor) {
  const double configured_limit =
      rk45_step_from_base(rk45_dt_max, scale_factor);
  const double lattice_omega_max =
      2.0 * std::sqrt(3.0) * std::pow(scale_factor, -rescale_s) / dx;
  const double phase_accuracy_limit =
      rk45_max_lattice_phase / lattice_omega_max;
  return std::min(configured_limit, phase_accuracy_limit);
}

static double rk45_step_to_scale_factor(double target_scale_factor) {
  const double delta_a = target_scale_factor - a;
  if (!(delta_a > 0.0) || !(ad > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }

  const double discriminant = ad * ad + 2.0 * ad2 * delta_a;
  if (std::isfinite(discriminant) && discriminant >= 0.0) {
    const double denominator = ad + std::sqrt(discriminant);
    if (denominator > 0.0) {
      return 2.0 * delta_a / denominator;
    }
  }
  return delta_a / ad;
}

static void validate_rk45_parameters() {
  const bool valid =
      std::isfinite(dt) && dt > 0.0
      && std::isfinite(rk45_rtol) && rk45_rtol > 0.0
      && std::isfinite(rk45_atol) && rk45_atol > 0.0
      && std::isfinite(rk45_geometry_atol) && rk45_geometry_atol > 0.0
      && std::isfinite(rk45_dt_min) && rk45_dt_min > 0.0
      && std::isfinite(rk45_dt_max) && rk45_dt_max >= rk45_dt_min
      && std::isfinite(rk45_safety)
      && rk45_safety > 0.0 && rk45_safety < 1.0
      && std::isfinite(rk45_min_factor)
      && rk45_min_factor > 0.0 && rk45_min_factor < 1.0
      && std::isfinite(rk45_max_factor) && rk45_max_factor >= 1.0
      && std::isfinite(rk45_max_lattice_phase)
      && rk45_max_lattice_phase > 0.0
      && rk45_max_lattice_phase < 0.997;
  if (!valid) {
    fprintf(stderr, "Invalid RK45 parameters in parameters.h.\n");
    exit(1);
  }
}

void run_evolution_loop(FILE* output_) {
  int numsteps = 0;
  simulation_initial_a = a;
  const auto inflaton_mean = []() {
    double mean = 0.0;
    for (int ii = 0; ii < N; ++ii)
      for (int jj = 0; jj < N; ++jj)
        for (int kk = 0; kk < N; ++kk)
          mean += f[idx_mf(0, ii, jj, kk)];
    return mean / static_cast<double>(gridsize);
  };
  double previous_phi_mean = inflaton_mean();
  double previous_efolds = 0.0;
  if (previous_phi_mean <= phi_c) {
    critical_crossing_efolds = 0.0;
  }

  const auto record_critical_crossing = [&]() {
    const double current_phi_mean = inflaton_mean();
    const double current_efolds = log(a / simulation_initial_a);
    if (!std::isfinite(critical_crossing_efolds)
        && previous_phi_mean > phi_c && current_phi_mean <= phi_c) {
      const double fraction = (previous_phi_mean - phi_c)
          / (previous_phi_mean - current_phi_mean);
      critical_crossing_efolds = previous_efolds
          + fraction * (current_efolds - previous_efolds);
    }
    previous_phi_mean = current_phi_mean;
    previous_efolds = current_efolds;
  };

  validate_rk45_parameters();
  {
    RK45Workspace workspace(f.size());
    double step = rk45_step_from_base(dt, a);
    long long rejected_steps = 0;
    double min_accepted_step = std::numeric_limits<double>::infinity();
    double max_accepted_step = 0.0;
    double min_accepted_base_step = std::numeric_limits<double>::infinity();
    double max_accepted_base_step = 0.0;
    double max_accepted_error = 0.0;


    const double endpoint_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(af));
    while (a < af - endpoint_tolerance) {
      const double a_before = a;
      const double minimum_step = rk45_minimum_step(a_before);
      const double maximum_step = rk45_maximum_step(a_before);
      if (!std::isfinite(minimum_step) || !std::isfinite(maximum_step)
          || maximum_step < minimum_step) {
        fprintf(stderr,
                "RK45 step bounds are inconsistent at a=%.17g: "
                "h_min=%.17g, h_max=%.17g.\n",
                a_before, minimum_step, maximum_step);
        exit(1);
      }
      const double endpoint_step = rk45_step_to_scale_factor(af);
      const bool endpoint_below_minimum =
          endpoint_step < minimum_step;
      const double controller_minimum =
          endpoint_below_minimum ? 0.0 : minimum_step;
      step = std::clamp(step, controller_minimum, maximum_step);
      const double attempted_step = std::min(step, endpoint_step);
      if (!(attempted_step > 0.0) || !std::isfinite(attempted_step)) {
        fprintf(stderr,
                "RK45 produced an invalid attempted step at a=%.17g: h=%.17g.\n",
                a_before, attempted_step);
        exit(1);
      }
      const double attempted_base_step = attempted_step
          * std::pow(a_before, 1.0 - rescale_s);

      const RK45StepResult result = rk45_evolve(attempted_step, workspace);
      if (!result.accepted) {
        rejected_steps++;
        const double proposed_step = std::isfinite(result.suggested_step)
            ? result.suggested_step
            : attempted_step * rk45_min_factor;
        const bool no_smaller_step_exists =
            !(proposed_step > 0.0) || !(proposed_step < attempted_step);
        if (!endpoint_below_minimum
            && attempted_step <= minimum_step
                * (1.0 + 32.0 * std::numeric_limits<double>::epsilon())) {
          fprintf(stderr,
                  "RK45 cannot satisfy its tolerance at the minimum step: "
                  "a=%.17g, h=%.17g, error=%g.\n",
                  a_before, attempted_step, result.error_norm);
          exit(1);
        }
        if (endpoint_below_minimum && no_smaller_step_exists) {
          fprintf(stderr,
                  "RK45 cannot resolve the final scale-factor step: "
                  "a=%.17g, h=%.17g, error=%g.\n",
                  a_before, attempted_step, result.error_norm);
          exit(1);
        }
        step = proposed_step;
        // The state, including a, is unchanged after rejection.
        step = std::clamp(step, controller_minimum, maximum_step);
        continue;
      }

      numsteps++;
      min_accepted_step = std::min(min_accepted_step, attempted_step);
      max_accepted_step = std::max(max_accepted_step, attempted_step);
      min_accepted_base_step =
          std::min(min_accepted_base_step, attempted_base_step);
      max_accepted_base_step =
          std::max(max_accepted_base_step, attempted_base_step);
      max_accepted_error = std::max(max_accepted_error, result.error_norm);
      astep = a;
      record_critical_crossing();

      const double next_minimum_step = rk45_minimum_step(a);
      const double next_maximum_step = rk45_maximum_step(a);
      if (next_maximum_step < next_minimum_step) {
        fprintf(stderr,
                "RK45 step bounds became inconsistent at a=%.17g: "
                "h_min=%.17g, h_max=%.17g.\n",
                a, next_minimum_step, next_maximum_step);
        exit(1);
      }
      step = std::clamp(result.suggested_step,
                        next_minimum_step, next_maximum_step);

      if (numsteps % output_freq == 0 && a < af) {
        save((numsteps % output_infrequent_freq == 0) ? 1 : 0);
      }

      if (numsteps % output_freq == 0) {
        if (screen_updates) {
          printf("scale factor a = %.16e\n",
                 a);
          printf("numsteps %i\n\n", numsteps);
        }
        fprintf(output_,
                "scale factor a = %.16e\n", a);
        fprintf(output_, "numsteps %i\n\n", numsteps);
        fflush(output_);
      }

    }

    if (numsteps == 0) {
      min_accepted_step = 0.0;
      min_accepted_base_step = 0.0;
    }
  } 

  printf("Saving final inflaton data\n");
  save(1);
  save_last();
}


// -------------------- DeltaN Evolution --------------------

static std::vector<unsigned char> deltaN_active;
static long long deltaN_active_count = 0;
static constexpr int deltaN_waterfall_field = (nFields > 1) ? 1 : 0;

struct DeltaNHandoffDiagnostics {
  float laplacian_waterfall_over_a2;
  float gradient_energy_all_fields;
  float kinetic_energy_all_fields;
  float potential_energy;
  float gradient_acceleration_waterfall;
  float potential_acceleration_waterfall;
  float hubble_drag_waterfall;
  float gradient_force_ratio;
};

static inline double forward_derivative_mf(
    int field_index, int direction, int i, int j, int k) {
  if (direction == 0) {
    return (f[idx_mf(field_index,INCREMENT(i),j,k)]
          - f[idx_mf(field_index,i,j,k)]) / dx;
  }
  if (direction == 1) {
    return (f[idx_mf(field_index,i,INCREMENT(j),k)]
          - f[idx_mf(field_index,i,j,k)]) / dx;
  }
  return (f[idx_mf(field_index,i,j,INCREMENT(k))]
        - f[idx_mf(field_index,i,j,k)]) / dx;
}

static std::vector<DeltaNHandoffDiagnostics>
calculate_deltaN_handoff_diagnostics() {
  std::vector<DeltaNHandoffDiagnostics> diagnostics(gridsize);
  const double H_code = ad * std::pow(a, rescale_s - 2.0);
  const double H2 = pw2(H_code);
  if (!(H2 > 0.0) || !std::isfinite(H2)) {
    printf("Invalid Hubble rate while calculating deltaN handoff diagnostics\n");
    exit(1);
  }

#if parallel_calculation
#pragma omp parallel
#endif
  {
    std::vector<double> fields(nFields, 0.0);
#if parallel_calculation
#pragma omp for collapse(3)
#endif
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
        const size_t id = idx(i,j,k);
        double gradient_sq = 0.0;
        double derivative_norm_sq = 0.0;
        for (int field_index = 0; field_index < nFields; ++field_index) {
          fields[field_index] = f[idx_mf(field_index,i,j,k)];
          derivative_norm_sq += pw2(fd[idx_mf(field_index,i,j,k)]);
          for (int direction = 0; direction < 3; ++direction) {
            gradient_sq += pw2(forward_derivative_mf(
                field_index, direction, i, j, k));
          }
        }

        const double rho_gradient = 0.5 * gradient_sq / pw2(a);
        const double rho_kinetic = 0.5 * H2 * derivative_norm_sq;
        const double rho_potential = potential(fields);

        const double laplacian_over_a2 =
            lapl(i, j, k, f, deltaN_waterfall_field) / (pw2(a) * pw2(dx));
        const double gradient_acceleration = std::abs(laplacian_over_a2) / H2;
        const double potential_acceleration = std::abs(
            potential_derivative_at_state(fields, deltaN_waterfall_field)) / H2;
        const double epsilon_local = 0.5 * derivative_norm_sq;
        const double hubble_drag = std::abs(
            (3.0 - epsilon_local)
            * fd[idx_mf(deltaN_waterfall_field,i,j,k)]);
        const double force_denominator =
            potential_acceleration + hubble_drag;

        diagnostics[id] = {
            static_cast<float>(laplacian_over_a2),
            static_cast<float>(rho_gradient),
            static_cast<float>(rho_kinetic),
            static_cast<float>(rho_potential),
            static_cast<float>(gradient_acceleration),
            static_cast<float>(potential_acceleration),
            static_cast<float>(hubble_drag),
            static_cast<float>(force_denominator > 0.0
                ? gradient_acceleration / force_denominator
                : std::numeric_limits<double>::infinity())};
        }
      }
    }
  }
  return diagnostics;
}

static void write_deltaN_handoff_scatter(
    const std::vector<double>& handoff_sigma,
    const std::vector<double>& handoff_dsigma_dN,
    const std::vector<DeltaNHandoffDiagnostics>& diagnostics) {
  const std::string filename = path + "/deltaN_handoff_scatter" + ext_;
  FILE* scatter_file = fopen(filename.c_str(), "w");
  if (!scatter_file) {
    printf("Could not open %s for writing\n", filename.c_str());
    exit(1);
  }

  fprintf(scatter_file,
          "# sigma_handoff dsigma_dN_handoff deltaN "
          "laplacian_sigma_over_a2 gradient_energy_all_fields "
          "kinetic_energy_all_fields potential_energy rho_total "
          "gradient_over_total gradient_over_kinetic "
          "gradient_acceleration_sigma potential_acceleration_sigma "
          "hubble_drag_sigma gradient_force_ratio\n");
  for (size_t id = 0; id < deltaN.size(); ++id) {
    const DeltaNHandoffDiagnostics& d = diagnostics[id];
    const double rho_total = d.gradient_energy_all_fields
        + d.kinetic_energy_all_fields + d.potential_energy;
    const double gradient_over_total = rho_total != 0.0
        ? d.gradient_energy_all_fields / rho_total
        : std::numeric_limits<double>::quiet_NaN();
    const double gradient_over_kinetic = d.kinetic_energy_all_fields != 0.0
        ? d.gradient_energy_all_fields / d.kinetic_energy_all_fields
        : std::numeric_limits<double>::infinity();
    fprintf(scatter_file,
            "%.17e %.17e %.17e %.9e %.9e %.9e %.9e %.9e %.9e %.9e "
            "%.9e %.9e %.9e %.9e\n",
            handoff_sigma[id], handoff_dsigma_dN[id], deltaN[id],
            d.laplacian_waterfall_over_a2,
            d.gradient_energy_all_fields,
            d.kinetic_energy_all_fields,
            d.potential_energy,
            rho_total,
            gradient_over_total,
            gradient_over_kinetic,
            d.gradient_acceleration_waterfall,
            d.potential_acceleration_waterfall,
            d.hubble_drag_waterfall,
            d.gradient_force_ratio);
  }
  fclose(scatter_file);
}
static inline double ref_potential() {
  return potential(phiref);
}

static std::vector<double> mean_field_values() {
  std::vector<double> mean_fields(nFields, 0.0);
  DECLARE_INDICES

  LOOP_MF {
    LOOP {
      mean_fields[n] += f[idx_mf(n,i,j,k)];
    }
    mean_fields[n] /= (double)gridsize;
  }

  return mean_fields;
}

static std::vector<double> mean_field_derivatives() {
  std::vector<double> mean_derivatives(nFields, 0.0);
  DECLARE_INDICES

  LOOP_MF {
    LOOP {
      mean_derivatives[n] += fd[idx_mf(n,i,j,k)];
    }
    mean_derivatives[n] /= (double)gridsize;
  }

  return mean_derivatives;
}

static double end_of_inflation_margin(const std::vector<double>& field_values,
                                      double& epsilon_value,
                                      double& waterfall_eta_value) {
  const double V = potential(field_values);
  if (!(V > 0.0) || !std::isfinite(V)) {
    printf("Invalid potential while locating the deltaN end surface: V = %e\n", V);
    exit(1);
  }

  epsilon_value = 0.0;
  for (int field_index = 0; field_index < nFields; ++field_index) {
    const double dV = potential_derivative_at_state(field_values, field_index);
    epsilon_value += 0.5 * pw2(dV / V);
  }

  waterfall_eta_value = eta(field_values, deltaN_waterfall_field);
  return deltaN_waterfall_eta_end - waterfall_eta_value;
}

static std::vector<double> deltaN_acceleration(
    const std::vector<double>& field_values,
    const std::vector<double>& field_derivatives) {
  double derivative_norm_sq = 0.0;
  for (double derivative : field_derivatives) {
    derivative_norm_sq += pw2(derivative);
  }

  const double V = potential(field_values);
  if (!(V > 0.0) || !std::isfinite(V)) {
    printf("Invalid potential in deltaN evolution: V = %e\n", V);
    exit(1);
  }

  const double friction = 3.0 - 0.5 * derivative_norm_sq;
  std::vector<double> acceleration(nFields, 0.0);
  for (int field_index = 0; field_index < nFields; ++field_index) {
    const double potential_ratio =
        potential_derivative_at_state(field_values, field_index) / V;
    acceleration[field_index] =
        -friction * (field_derivatives[field_index] + potential_ratio);
  }
  return acceleration;
}

static std::vector<double> get_phiref_end_of_inflation(
    double& epsilon_ref,
    double& waterfall_eta_ref,
    double& deltaN_ref) {
  std::vector<double> background_fields = mean_field_values();
  std::vector<double> background_derivs = mean_field_derivatives();
  deltaN_ref = 0.0;

  double margin = end_of_inflation_margin(
      background_fields, epsilon_ref, waterfall_eta_ref);
  if (margin >= 0.0) {
    phiref = background_fields;
    return background_fields;
  }

  std::vector<double> acceleration =
      deltaN_acceleration(background_fields, background_derivs);
  for (int field_index = 0; field_index < nFields; ++field_index) {
    background_derivs[field_index] += 0.5 * dN * acceleration[field_index];
  }

  while (deltaN_ref < Nend) {
    const double step = std::min(dN, Nend - deltaN_ref);
    const std::vector<double> fields_before = background_fields;
    const double margin_before = margin;
    for (int field_index = 0; field_index < nFields; ++field_index) {
      background_fields[field_index] += step * background_derivs[field_index];
    }

    margin = end_of_inflation_margin(
        background_fields, epsilon_ref, waterfall_eta_ref);
    if (margin >= 0.0) {
      double crossing_fraction = 1.0;
      const double margin_change = margin - margin_before;
      if (margin_before < 0.0 && margin_change > 0.0) {
        crossing_fraction = std::clamp(-margin_before / margin_change, 0.0, 1.0);
      }
      for (int field_index = 0; field_index < nFields; ++field_index) {
        background_fields[field_index] = fields_before[field_index]
            + crossing_fraction * step * background_derivs[field_index];
      }
      deltaN_ref += crossing_fraction * step;
      end_of_inflation_margin(
          background_fields, epsilon_ref, waterfall_eta_ref);
      phiref = background_fields;
      return background_fields;
    }

    deltaN_ref += step;
    acceleration = deltaN_acceleration(background_fields, background_derivs);
    for (int field_index = 0; field_index < nFields; ++field_index) {
      background_derivs[field_index] += step * acceleration[field_index];
    }
  }

  printf("Background did not reach the end of inflation by deltaN Nend = %f "
         "(epsilon = %e, waterfall eta = %e).\n",
         Nend, epsilon_ref, waterfall_eta_ref);
  exit(1);
}

static inline double deltaN_surface_value(
    const std::vector<double>& field_values) {
  return eta(field_values, deltaN_waterfall_field)
      - deltaN_waterfall_eta_end;
}

static inline void load_deltaN_patch_fields(
    int i, int j, int k, std::vector<double>& field_values) {
  for (int field_index = 0; field_index < nFields; ++field_index) {
    field_values[field_index] = f[idx_mf(field_index,i,j,k)];
  }
}

static inline double deltaN_surface_value_at(int i, int j, int k) {
  std::vector<double> field_values(nFields, 0.0);
  load_deltaN_patch_fields(i, j, k, field_values);
  return deltaN_surface_value(field_values);
}

static void initialize_deltaN_active_cells() {
  deltaN_active.assign(gridsize, 0);
  deltaN_active_count = 0;
  int i, j, k;
  LOOP {
    const size_t id = idx(i,j,k);
    if (deltaN_surface_value_at(i,j,k) > 0.0) {
      deltaN_active[id] = 1;
      deltaN_active_count++;
    }
  }

  if (deltaN_active_count != gridsize) {
    printf("Warning: %lld deltaN patches are already at or beyond the reference surface.\n",
           static_cast<long long>(gridsize) - deltaN_active_count);
  }
}

// Update fd in e-folding time coordinates
void evolve_derivsN(double d) {
  int i, j, k;
#if parallel_calculation
#pragma omp parallel for collapse(3)
#endif
  LOOP {
    if (!deltaN_active[idx(i,j,k)]) continue;

    double fd_norm_sq = 0.0;
    for (int field_index = 0; field_index < nFields; ++field_index) {
      fd_norm_sq += pw2(fd[idx_mf(field_index,i,j,k)]);
    }

    const double friction = (3.0 - 0.5 * fd_norm_sq);
    for (int field_index = 0; field_index < nFields; ++field_index) {
      fd[idx_mf(field_index,i,j,k)] += d * (
          -friction * (fd[idx_mf(field_index,i,j,k)] + pot_ratio(field_index, i, j, k)));
    }
  }
}

// Update f and deltaN grid
void evolve_fieldsN(double d) {
  int i, j, k;
  long long stopped_this_step = 0;
#if parallel_calculation
#pragma omp parallel for collapse(3) reduction(+:stopped_this_step)
#endif
  LOOP {
    const size_t id = idx(i,j,k);
    if (!deltaN_active[id]) continue;

    std::vector<double> patch_fields(nFields, 0.0);
    load_deltaN_patch_fields(i, j, k, patch_fields);
    const double surface_before = deltaN_surface_value(patch_fields);
    for (int field_index = 0; field_index < nFields; ++field_index) {
      f[idx_mf(field_index,i,j,k)] += d * fd[idx_mf(field_index,i,j,k)];
    }
    load_deltaN_patch_fields(i, j, k, patch_fields);
    const double surface_after = deltaN_surface_value(patch_fields);

    if (surface_after <= 0.0) {
      double lower_fraction = 0.0;
      double upper_fraction = 1.0;
      if (surface_before > 0.0) {
        for (int refinement = 0; refinement < 20; ++refinement) {
          const double trial_fraction =
              0.5 * (lower_fraction + upper_fraction);
          for (int field_index = 0; field_index < nFields; ++field_index) {
            patch_fields[field_index] =
                f[idx_mf(field_index,i,j,k)]
                - (1.0 - trial_fraction) * d
                  * fd[idx_mf(field_index,i,j,k)];
          }
          if (deltaN_surface_value(patch_fields) > 0.0) {
            lower_fraction = trial_fraction;
          } else {
            upper_fraction = trial_fraction;
          }
        }
      }
      const double crossing_fraction =
          0.5 * (lower_fraction + upper_fraction);

      // Put the patch exactly on the interpolated reference hypersurface
      for (int field_index = 0; field_index < nFields; ++field_index) {
        f[idx_mf(field_index,i,j,k)] -=
            (1.0 - crossing_fraction) * d * fd[idx_mf(field_index,i,j,k)];
      }
      deltaN[id] += crossing_fraction * d;
      deltaN_active[id] = 0;
      stopped_this_step++;
    } else {
      deltaN[id] += d;
    }
  }
  deltaN_active_count -= stopped_this_step;
}

// Run main deltaN integration loop
void run_deltaN_loop(FILE* output_) {
  printf("Starting deltaN calculation\n");
  fprintf(output_, "Starting deltaN calculation\n");

  int numsteps = 0;
  Ne = 0.0;

  if (deltaN_apply_smoothing) {
    smooth_all_fields_for_deltaN();
  } else {
    printf("deltaN handoff: Gaussian smoothing disabled; using all lattice modes\n");
    fprintf(output_,
            "deltaN handoff: Gaussian smoothing disabled; using all lattice modes\n");
  }
  initializeN();

  std::vector<double> handoff_sigma(gridsize, 0.0);
  std::vector<double> handoff_dsigma_dN(gridsize, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        const size_t id = idx(i,j,k);
        handoff_sigma[id] = f[idx_mf(deltaN_waterfall_field,i,j,k)];
        handoff_dsigma_dN[id] =
            fd[idx_mf(deltaN_waterfall_field,i,j,k)];
      }
    }
  }
  const std::vector<DeltaNHandoffDiagnostics> handoff_diagnostics =
      calculate_deltaN_handoff_diagnostics();

  double epsilon_ref = 0.0;
  double waterfall_eta_ref = 0.0;
  double deltaN_ref = 0.0;
  phiref = get_phiref_end_of_inflation(
      epsilon_ref, waterfall_eta_ref, deltaN_ref);
  deltaN_end_surface_N = deltaN_ref;
  printf("Background end after ΔN = %f\n",
         deltaN_ref);
  fprintf(output_, "Background end after ΔN = %f\n",
          deltaN_ref);

  initialize_deltaN_active_cells();
  evolve_derivsN(0.5 * dN);

  while (deltaN_active_count > 0 && Ne < Nend) {
    const double step = std::min(dN, Nend - Ne);
    evolve_fieldsN(step);
    Ne += step;
    numsteps++;

    if (deltaN_active_count > 0) {
      evolve_derivsN(step);
    }

    if (screen_updates && numsteps % output_freq == 0) {
      printf("deltaN N = %f, active patches = %lld\n\n",
             Ne, deltaN_active_count);
    }

    if (numsteps % output_freq == 0) {
      fprintf(output_, "deltaN N = %f, active patches = %lld\n\n",
              Ne, deltaN_active_count);
      fflush(output_);
    }
  }

  if (deltaN_active_count > 0) {
    printf("deltaN failed: %lld patches did not reach the reference surface by Nend = %f\n",
           deltaN_active_count, Nend);
    fprintf(output_, "deltaN failed: %lld patches did not reach the reference surface by Nend = %f\n",
            deltaN_active_count, Nend);
    fflush(output_);
    exit(1);
  }

  double mean_deltaN = 0.0;
  for (double patch_deltaN : deltaN) {
    mean_deltaN += patch_deltaN;
  }
  mean_deltaN /= static_cast<double>(deltaN.size());

  printf("All deltaN patches reached eta_%d = %.6f by N = %f; "
         "mean patch deltaN = %.6f\n",
         deltaN_waterfall_field, deltaN_waterfall_eta_end, Ne,
         mean_deltaN);
  fprintf(output_,
          "All deltaN patches reached eta_%d = %.6f by N = %f; "
          "mean patch deltaN = %.17g\n",
          deltaN_waterfall_field, deltaN_waterfall_eta_end, Ne,
          mean_deltaN);
  fflush(output_);
  write_deltaN_handoff_scatter(
      handoff_sigma, handoff_dsigma_dN, handoff_diagnostics);
  saveN();
}
