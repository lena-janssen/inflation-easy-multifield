// initialize.cpp - Initialization routines for field values and simulation parameters

#include "main.h"
#include "ffteasy.hpp"

// -------------------- Random Number Generator --------------------
#define randa 16807
#define randm 2147483647
#define randq 127773
#define randr 2836

double rand_uniform(void) {
  if (seed < 1) return 0.33; // Fixed fallback for debugging
  static int i = 0;
  static int next = seed;
  if (!(next > 0)) {
    printf("Invalid seed used in random number function. Using seed=1\n");
    next = 1;
  }
  if (i == 0) for (i = 1; i < 100; i++) rand_uniform();

  next = randa * (next % randq) - randr * (next / randq);
  if (next < 0) next += randm;
  return (static_cast<double>(next) / static_cast<double>(randm));
}

#undef randa
#undef randm
#undef randq
#undef randr

// -------------------- High-Level Simulation Initialization --------------------

void initialize_simulation() {

  printf("Path name for this run: %s\n", path.c_str());
  printf("L/N = %f\n", L / N);

  printf("Critical field value for waterfall transition: phi_crit = %.8e\n", phi_c);
  printf("Global minima of potential at phi=0, sigma=±M=%e: V = %.8e\n", mass[2], potential({0.0, mass[2]}));

  initialize();        // Basic checks and global param setup
  initializef();       // Set field configuration and apply FFT
  output_parameters(); // Save run configuration
  save(1);             // First output
  t = t0;              // Set initial time
}

// -------------------- Field Mode Initialization --------------------
// Set amplitude and phase of vacuum mode using Rayleigh distribution
void set_mode(double p2, double p2lat, double m2, double *field, double *deriv, int real) {
    double phase, amplitude, rms_amplitude, omega;
    double re_f_left, im_f_left;
    double norm = rescale_B * pow(L / pw2(dx), 1.5); // Normalization (see 2209.13616)
    double ii = L / (2.0 * pi) * sqrt(p2lat);
    double k_mag = sqrt(p2lat);
    const double k_fundamental = 2.0 * pi / L;
    double hbterm = -hubble_init;
    static int tachyonic = 0;
    static int warned_invalid_k_cutoff = 0;

    omega = (p2 + m2 > 0.0) ? sqrt(p2 + m2) : sqrt(p2);
    if (p2 + m2 <= 0.0 && tachyonic == 0) {
        printf("Warning: Tachyonic mode(s) may be initialized inaccurately\n");
        tachyonic = 1;
    }

    rms_amplitude = (omega > 0.0) ? norm / sqrt(2.0 * omega) : 0.0;

    static long kept = 0;
    static long cut = 0;

    bool is_cut = false;
    bool is_index_cut = false;
    bool is_k_cut = false;

    if (high_cutoff_index > 0.0 &&
        (ii > high_cutoff_index || ii < low_cutoff_index)) {
      is_index_cut = true;
    }

    const bool physical_cutoff_usable = (k_uv_cutoff <= 0.0 || k_uv_cutoff >= k_fundamental);
    if (!physical_cutoff_usable && !warned_invalid_k_cutoff) {
      printf("Warning: k_uv_cutoff = %.6e is below first nonzero lattice mode k1 = 2pi/L = %.6e. Ignoring physical cutoff to avoid zeroing all modes.\n",
           k_uv_cutoff, k_fundamental);
      warned_invalid_k_cutoff = 1;
    }

    if (k_uv_cutoff > 0.0 && physical_cutoff_usable &&
      (k_mag > k_uv_cutoff || (k_ir_cutoff > 0.0 && k_mag < k_ir_cutoff))) {
      is_k_cut = true;
    }

    if (is_index_cut || is_k_cut) {
      rms_amplitude = 0.0;
      is_cut = true;
    }

    if (is_cut) cut++;
    else kept++;

    if ((kept + cut) % 1000 == 0) {
        // printf("cutoff stats: kept=%ld cut=%ld last ii=%.6e rms=%.16e\n",
              // kept, cut, ii, rms_amplitude);
    }

    // printf("low=%.6e high=%.6e ii=%.6e\n",
    //    low_cutoff_index, high_cutoff_index, ii);

    amplitude = rms_amplitude * sqrt(log(1.0 / rand_uniform()));
    phase = 2.0 * pi * rand_uniform();

    re_f_left = amplitude * cos(phase);
    im_f_left = amplitude * sin(phase);

    field[0] = re_f_left;
    field[1] = im_f_left;
    deriv[0] = omega * im_f_left + hbterm * field[0];
    deriv[1] = -omega * re_f_left + hbterm * field[1];

    if (real == 1) {
        field[1] = 0.0;
        deriv[1] = 0.0;
    }
}

// -------------------- Effective Mass Calculation --------------------
std::vector<double> effective_mass_sq(const std::vector<double>& field_value)
  {
      const double phi = field_value[0];
      const double sigma = field_value[1];

      const double mu2 = mass[1];
      const double M   = mass[2];

      std::vector<double> m2(nFields, 0.0);

      // d²V/dphi²
      m2[0] = Lambda4 * (
          -2.0 / pw2(mu2)
          + 4.0 * pw2(sigma) / (pw2(phi_c) * pw2(M))
      ) / pw2(rescale_B);

      // d²V/dpsi²
      if (nFields > 1) {
          m2[1] = Lambda4 * (
              -4.0 / pw2(M)
              + 12.0 * pw2(sigma) / pw2(pw2(M))
              + 4.0 * pw2(phi) / (pw2(phi_c) * pw2(M))
          ) / pw2(rescale_B);
      }

      return m2;
  }

// -------------------- Basic Global Initialization --------------------

void initialize() {

  printf("Generating initial conditions for new run at t = 0\n");

  t0 = 0.0;
  hubble_init = sqrt(potential(initial_field) / 3.0);
  printf("Initial Hubble parameter = %f\n", hubble_init);
  printf("Initial horizon size in code units = %f\n", 1.0 / hubble_init);

  if (hubble_init < 0.0) {
    printf("Error: Initial Hubble constant is negative or undefined\n");
    exit(1);
  }

  ad = hubble_init;

}

// -------------------- Vacuum Fluctuation Initialization --------------------

// Used to compute the initial background field derivatives
static std::vector<double> homogeneous_acceleration_N(
    const std::vector<double>& fields,
    const std::vector<double>& efold_derivatives) {
  const double V = potential(fields);
  if (!(V > 0.0) || !std::isfinite(V)) {
    printf("Invalid potential in initial-attractor precursor: V=%e\n", V);
    exit(1);
  }

  double derivative_norm_sq = 0.0;
  for (double derivative : efold_derivatives) {
    derivative_norm_sq += derivative * derivative;
  }
  const double friction = 3.0 - 0.5 * derivative_norm_sq;
  if (!(friction > 0.0) || !std::isfinite(friction)) {
    printf("Invalid kinetic state in initial-attractor precursor.\n");
    exit(1);
  }

  std::vector<double> acceleration(nFields, 0.0);
  for (int field_index = 0; field_index < nFields; ++field_index) {
    acceleration[field_index] = -friction * (
        efold_derivatives[field_index]
        + potential_derivative_at_state(fields, field_index) / V);
  }
  return acceleration;
}

static void rk4_homogeneous_N_step(
    std::vector<double>& fields,
    std::vector<double>& efold_derivatives,
    double step) {
  const std::vector<double> k1_fields = efold_derivatives;
  const std::vector<double> k1_derivatives =
      homogeneous_acceleration_N(fields, efold_derivatives);

  std::vector<double> stage_fields(nFields, 0.0);
  std::vector<double> stage_derivatives(nFields, 0.0);
  for (int field_index = 0; field_index < nFields; ++field_index) {
    stage_fields[field_index] = fields[field_index]
        + 0.5 * step * k1_fields[field_index];
    stage_derivatives[field_index] = efold_derivatives[field_index]
        + 0.5 * step * k1_derivatives[field_index];
  }
  const std::vector<double> k2_fields = stage_derivatives;
  const std::vector<double> k2_derivatives =
      homogeneous_acceleration_N(stage_fields, stage_derivatives);

  for (int field_index = 0; field_index < nFields; ++field_index) {
    stage_fields[field_index] = fields[field_index]
        + 0.5 * step * k2_fields[field_index];
    stage_derivatives[field_index] = efold_derivatives[field_index]
        + 0.5 * step * k2_derivatives[field_index];
  }
  const std::vector<double> k3_fields = stage_derivatives;
  const std::vector<double> k3_derivatives =
      homogeneous_acceleration_N(stage_fields, stage_derivatives);

  for (int field_index = 0; field_index < nFields; ++field_index) {
    stage_fields[field_index] = fields[field_index]
        + step * k3_fields[field_index];
    stage_derivatives[field_index] = efold_derivatives[field_index]
        + step * k3_derivatives[field_index];
  }
  const std::vector<double> k4_fields = stage_derivatives;
  const std::vector<double> k4_derivatives =
      homogeneous_acceleration_N(stage_fields, stage_derivatives);

  for (int field_index = 0; field_index < nFields; ++field_index) {
    fields[field_index] += step / 6.0 * (
        k1_fields[field_index] + 2.0 * k2_fields[field_index]
        + 2.0 * k3_fields[field_index] + k4_fields[field_index]);
    efold_derivatives[field_index] += step / 6.0 * (
        k1_derivatives[field_index] + 2.0 * k2_derivatives[field_index]
        + 2.0 * k3_derivatives[field_index] + k4_derivatives[field_index]);
  }
}

static void determine_initial_background_derivative() {
  initial_field_derivative_used = initial_field_derivative;
  initial_field_efold_derivative_used.assign(nFields, 0.0);

  if (initialize_background_on_attractor) {
    if (!(initial_attractor_relaxation_efolds > 0.0)
        || !(initial_attractor_step > 0.0)
        || !std::isfinite(initial_attractor_relaxation_efolds)
        || !std::isfinite(initial_attractor_step)) {
      printf("Invalid initial-attractor precursor parameters.\n");
      exit(1);
    }

    std::vector<double> precursor_fields = initial_field;
    precursor_fields[0] += initial_attractor_relaxation_efolds / mass[0];
    std::vector<double> precursor_derivatives(nFields, 0.0);
    const double precursor_potential = potential(precursor_fields);
    for (int field_index = 0; field_index < nFields; ++field_index) {
      precursor_derivatives[field_index] =
          -potential_derivative_at_state(precursor_fields, field_index)
          / precursor_potential;
    }

    const long long max_steps = static_cast<long long>(std::ceil(
        2.0 * (initial_attractor_relaxation_efolds + 1.0)
        / initial_attractor_step));
    bool reached_target = false;
    double elapsed_efolds = 0.0;
    for (long long step_index = 0; step_index < max_steps; ++step_index) {
      const std::vector<double> fields_before = precursor_fields;
      const std::vector<double> derivatives_before = precursor_derivatives;
      rk4_homogeneous_N_step(
          precursor_fields, precursor_derivatives, initial_attractor_step);
      elapsed_efolds += initial_attractor_step;

      if (precursor_fields[0] <= initial_field[0]) {
        const double field_change =
            precursor_fields[0] - fields_before[0];
        double fraction = 1.0;
        if (field_change != 0.0) {
          fraction = (initial_field[0] - fields_before[0]) / field_change;
          fraction = std::max(0.0, std::min(1.0, fraction));
        }
        for (int field_index = 0; field_index < nFields; ++field_index) {
          precursor_derivatives[field_index] =
              derivatives_before[field_index]
              + fraction * (precursor_derivatives[field_index]
                            - derivatives_before[field_index]);
        }
        elapsed_efolds -= (1.0 - fraction) * initial_attractor_step;
        reached_target = true;
        break;
      }
    }

    if (!reached_target) {
      printf("Initial-attractor precursor did not reach phi_init.\n");
      exit(1);
    }
    initial_field_efold_derivative_used = precursor_derivatives;

    double derivative_norm_sq = 0.0;
    for (double derivative : initial_field_efold_derivative_used) {
      derivative_norm_sq += derivative * derivative;
    }
    const double denominator = 3.0 - 0.5 * derivative_norm_sq;
    const double background_hubble =
        std::sqrt(potential(initial_field) / denominator);
    for (int field_index = 0; field_index < nFields; ++field_index) {
      initial_field_derivative_used[field_index] =
          background_hubble * initial_field_efold_derivative_used[field_index];
    }

    printf("Initial homogeneous attractor obtained after %.6f precursor "
           "e-folds\n", elapsed_efolds);
  } else {
    double derivative_norm_sq = 0.0;
    for (double derivative : initial_field_derivative_used) {
      derivative_norm_sq += derivative * derivative;
    }
    const double background_hubble = std::sqrt(
        (potential(initial_field) + 0.5 * derivative_norm_sq) / 3.0);
    for (int field_index = 0; field_index < nFields; ++field_index) {
      initial_field_efold_derivative_used[field_index] =
          initial_field_derivative_used[field_index] / background_hubble;
    }
  }

  for (int field_index = 0; field_index < nFields; ++field_index) {
    printf("Initial background field %d: dphi/dN=%.16e, "
           "dphi/dtau=%.16e%s\n",
           field_index,
           initial_field_efold_derivative_used[field_index],
           initial_field_derivative_used[field_index],
           initialize_background_on_attractor ? " (attractor)" : " (input)");
  }
}

void initializef() {
  int n;
  f.resize(nFields * N * N * N, 0.0);
  fd.resize(nFields * N * N * N, 0.0);

  double p2, p2lat;
  double dp2 = pw2(2.0 * pi / L);
  int i, j, k, iconj, jconj;
  double px, py, pz;
  int arraysize[] = {N, N, N};


  ad = hubble_init;

  determine_initial_background_derivative();

  std::vector<double> eff_mass_sq = effective_mass_sq(initial_field);
  LOOP_MF {
  double mass_sq = eff_mass_sq[n];
  const bool initialize_field_modes = initialize_vacuum_fluctuations.at(n);
  for (i = 0; i < N; i++) {
    px = (i <= N / 2 ? i : i - N);
    iconj = (i == 0 ? 0 : N - i);

    for (j = 0; j < N; j++) {
      py = (j <= N / 2 ? j : j - N);

      for (k = 1; k < N / 2; k++) {
        pz = k;
        p2lat = dp2 * (pw2(px) + pw2(py) + pw2(pz));
        p2 = 4.0 * pw2(N / L) * (pw2(sin(px * pi / N)) + pw2(sin(py * pi / N)) + pw2(sin(pz * pi / N)));
        if (initialize_field_modes)
          set_mode(p2, p2lat, mass_sq, &f[idx_mf(n,i,j,2*k)], &fd[idx_mf(n,i,j,2*k)], 0);
      }

      if (j > N / 2 || (i > N / 2 && (j == 0 || j == N / 2))) {
        jconj = (j == 0 ? 0 : N - j);

        p2lat = dp2 * (pw2(px) + pw2(py));
        p2 = 4.0 * pw2(N / L) * (pw2(sin(px * pi / N)) + pw2(sin(py * pi / N)));
        if (initialize_field_modes)
          set_mode(p2, p2lat, mass_sq, &f[idx_mf(n,i,j,0)], &fd[idx_mf(n, i,j,0)], 0);

        f[idx_mf(n,iconj,jconj,0)] = f[idx_mf(n,i,j,0)];
        f[idx_mf(n,iconj,jconj,1)] = -f[idx_mf(n,i,j,1)];
        fd[idx_mf(n,iconj,jconj,0)] = fd[idx_mf(n,i,j,0)];
        fd[idx_mf(n,iconj,jconj,1)] = -fd[idx_mf(n,i,j,1)];

        p2lat = dp2 * (pw2(px) + pw2(py) + pw2(N / 2));
        p2 = 4.0 * pw2(N / L) * (pw2(sin(px * pi / N)) + pw2(sin(py * pi / N)) + 1.0);
        if (initialize_field_modes)
          set_mode(p2, p2lat, mass_sq, &fnyquist_p[n][i][2*j], &fdnyquist_p[n][i][2*j], 0);

        fnyquist_p[n][iconj][2*jconj]   = fnyquist_p[n][i][2*j];
        fnyquist_p[n][iconj][2*jconj+1] = -fnyquist_p[n][i][2*j+1];
        fdnyquist_p[n][iconj][2*jconj]   = fdnyquist_p[n][i][2*j];
        fdnyquist_p[n][iconj][2*jconj+1] = -fdnyquist_p[n][i][2*j+1];
      } else if ((i == 0 || i == N / 2) && (j == 0 || j == N / 2)) {
        p2lat = dp2 * (pw2(px) + pw2(py));
        p2 = 4.0 * pw2(N / L) * (pw2(sin(px * pi / N)) + pw2(sin(py * pi / N)));
        if (initialize_field_modes && p2 > 0.)
          set_mode(p2, p2lat, mass_sq, &f[idx_mf(n,i,j,0)], &fd[idx_mf(n,i,j,0)], 1);

        p2lat = dp2 * (pw2(px) + pw2(py) + pw2(N / 2));
        p2 = 4.0 * pw2(N / L) * (pw2(sin(px * pi / N)) + pw2(sin(py * pi / N)) + 1.0);
        if (initialize_field_modes)
          set_mode(p2, p2lat, mass_sq, &fnyquist_p[n][i][2*j], &fdnyquist_p[n][i][2*j], 1);
      }
    }
  }

  f[idx_mf(n,0,0,0)] = 0.;
  f[idx_mf(n,0,0,1)] = 0.;
  fd[idx_mf(n,0,0,0)] = 0.;
  fd[idx_mf(n,0,0,1)] = 0.;


  double* field_ptr = f.data() + n * N * N * N;
  double* deriv_ptr = fd.data() + n * N * N * N;

  fftrnd(field_ptr, (double*)fnyquist_p[n], 3, arraysize, -1);
  fftrnd(deriv_ptr, (double*)fdnyquist_p[n], 3, arraysize, -1);

  double rms = 0.0;

  LOOP {
      rms += pw2(f[idx_mf(n,i,j,k)]);
  }

  rms = sqrt(rms/(N*N*N));

  printf("field %d rms = %.16e\n", n, rms);
      LOOP {
    // f[idx_mf(n,i,j,k)] = 0.0;
    // fd[idx_mf(n,i,j,k)] = 0.0;
    f[idx_mf(n,i,j,k)] += initial_field[n];
    fd[idx_mf(n,i,j,k)] += initial_field_derivative_used[n];
  }
 }

  double vard = 0;
  LOOP_MF {
  LOOP vard += pw2(fd[idx_mf(n,i,j,k)]);
  }

  double deriv_energy_in = 0.5 * vard / static_cast<double>(gridsize);

  hubble_init = sqrt((deriv_energy_in + potential_energy(f) + gradient_energy(f, a)) / 3.);
  if (hubble_init < 0.0) {
    printf("Error in calculating initial Hubble constant. Exiting.\n");
    exit(1);
  }

  ad = hubble_init;
  printf("Finished initial conditions\n");
}

// -------------------- DeltaN Initialization --------------------

void smooth_deltaN_field(std::vector<double>& field, double cutoff_factor = 1.0) {
  int arraysize[] = {N, N, N};
  double (*field_nyquist)[2 * N] = fnyquist_p[0];

  const double dk = 2.0 * pi / L;
  const double kcut = cutoff_factor * ad * pow(a, rescale_s - 1.0);
  if (!(kcut > 0.0) || !std::isfinite(kcut)) {
    printf("Invalid deltaN smoothing scale kcut = %e\n", kcut);
    exit(1);
  }

  fftrnd(field.data(), (double*)field_nyquist, 3, arraysize, 1);

  for (int i = 0; i < N; ++i) {
    int px = (i <= N/2 ? i : i - N);
    for (int j = 0; j < N; ++j) {
      int py = (j <= N/2 ? j : j - N);

      for (int k = 1; k < N/2; ++k) {
        int pz = k;
        double kmag = dk * sqrt(px*px + py*py + pz*pz);

        double W = exp(-0.5 * pw2(kmag / kcut));  // Gaussian smoothing

        field[idx(i,j,2*k)]     *= W;
        field[idx(i,j,2*k + 1)] *= W;
      }

      for (int k = 0; k <= N/2; k += N/2) {
        int pz = k;
        double kmag = dk * sqrt(px*px + py*py + pz*pz);

        double W = exp(-0.5 * pw2(kmag / kcut));

        if (k == 0) {
          field[idx(i,j,0)] *= W;
          field[idx(i,j,1)] *= W;
        } else {
          field_nyquist[i][2*j]     *= W;
          field_nyquist[i][2*j + 1] *= W;
        }
      }
    }
  }

  fftrnd(field.data(), (double*)field_nyquist, 3, arraysize, -1);
}

void smooth_all_fields_for_deltaN() {
  const double comoving_hubble = ad * pow(a, rescale_s - 1.0);
  const double fundamental_mode = 2.0 * pi / L;
  const double fundamental_over_aH = fundamental_mode / comoving_hubble;

  printf("deltaN handoff: k_fund/(aH) = %.6e, smoothing k_cut/(aH) = %.6e\n",
         fundamental_over_aH, deltaN_smoothing_horizon_fraction);
  if (!(fundamental_over_aH > 0.0)
      || !std::isfinite(fundamental_over_aH)) {
    printf("Invalid deltaN handoff ratio k_fund/(aH) = %e\n",
           fundamental_over_aH);
    exit(1);
  }
  if (fundamental_over_aH > deltaN_max_fundamental_over_aH) {
    printf("deltaN handoff rejected: k_fund/(aH) = %.6e exceeds %.6e. "
           "Increase af or L so the separate-universe modes are safely "
           "superhorizon.\n",
           fundamental_over_aH, deltaN_max_fundamental_over_aH);
    exit(1);
  }

  DECLARE_INDICES
  for (n = 0; n < nFields; ++n) {
    std::vector<double> temp(N * N * N);

    LOOP {
      temp[idx(i,j,k)] = f[idx_mf(n,i,j,k)];
    }

    smooth_deltaN_field(temp, deltaN_smoothing_horizon_fraction);

    LOOP {
      f[idx_mf(n,i,j,k)] = temp[idx(i,j,k)];
    }
  }

  for (n = 0; n < nFields; ++n) {
    std::vector<double> temp(N * N * N);

    LOOP {
      temp[idx(i,j,k)] = fd[idx_mf(n,i,j,k)];
    }

    smooth_deltaN_field(temp, deltaN_smoothing_horizon_fraction);

    LOOP {
      fd[idx_mf(n,i,j,k)] = temp[idx(i,j,k)];
    }
  }
}
void initializeN() {
  DECLARE_INDICES
  deltaN.resize(N * N * N, 0.0);

  // Convert dphi/d\tilde{tau} to dphi/dN
  const double code_time_to_efolds = a / ad;
  if (!std::isfinite(code_time_to_efolds)) {
    printf("Invalid scale-factor derivative in deltaN initialization\n");
    exit(1);
  }

  LOOP {
    LOOP_MF {
      fd[idx_mf(n,i,j,k)] *= code_time_to_efolds;
    }
    deltaN[idx(i,j,k)] = 0.0;
  }
}
