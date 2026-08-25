// -------------------- InflationEasy Configuration Flags --------------------

// Set to 1 to perform deltaN evolution to calculate zeta
#define perform_deltaN 1

/// Set to 1 to enable OpenMP parallelism in selected loops
#define parallel_calculation 1


// -------------------- Lattice and Evolution Parameters --------------------

// Number of points along each spatial dimension (total lattice points will be N^3)
const int N = 64; // Must be a power of 2

// Rescaling exponent (must be 0 unless you've tested otherwise)
const double rescale_s = 0.0;

const double af = 1e4;

// RK45 integrator parameters
const double rk45_rtol = 1e-8;           // Relative local-error tolerance
const double rk45_atol = 1e-18;          // Absolute tolerance for fields/velocities
const double rk45_geometry_atol = 1e-12; // Absolute tolerance for a and da/dt
const double rk45_dt_max = 1e-3;
const double rk45_dt_min = 1e-9;
const double rk45_safety = 0.9;
const double rk45_min_factor = 0.2;
const double rk45_max_factor = 5.0;
const double rk45_max_lattice_phase = 0.25;

// Random seed for generating initial vacuum fluctuations
const int seed = 8;


// -------------------- Physical Model Parameters --------------------

// Parameters used for an analytic potential

const int nFields = 2; // Number of scalar fields in the simulation


// Hybrid inflation parameters

// Parameter setup
const std::vector<double> mass = {7.07e7, 10.7, 0.01}; // {mu1, mu2, M}
const double phi_c = sqrt(2) * 0.01;
const double Lambda4 = pow(2.66e-6, 4);

// Enter the path to the directory where the output files will be saved
const std::string path = "/results";

const double V0 = Lambda4;
const double rescale_B = sqrt(V0); // Rescaling factor (see doc)

const double PI = mass[2] * std::sqrt(mass[0] * phi_c);

const double initial_waterfall = std::sqrt( Lambda4 * PI / ( 48 * std::sqrt(2 * std::pow(M_PI, 3))) );  // Waterfall field sigma at the critical point, see [1501.07565] eq. (11)

const std::vector<double> initial_field = {phi_c + (1 / mass[0]), 0.0}; // Start 1 e-fold before the critical point

const bool initialize_background_on_attractor = true; // Determines the initial condition velocities from the background evolution
const double initial_attractor_relaxation_efolds = 5.0; // Number of e-folds to evolve the background before starting the lattice evolution
const double initial_attractor_step = 1.0e-3; // Step size for the initial attractor evolution
const std::vector<double> initial_field_derivative = {-7.738238530394281e-09, 0.0}; 

const std::vector<bool> initialize_vacuum_fluctuations = {true, true};

const double L = 1.0; // Box size in code units (must be sub-horizon at start)
const double dt = 0.001; // Initial base timestep for the adaptive RK45 integrator


// Output frequency in time steps (standard and infrequent quantities)
const int output_freq = 100;
const int output_infrequent_freq = 100;

const double dN = 0.0005; // e-fold increment in deltaN evolution
const double Nend = 30; // Safety cap; the run stops earlier once every patch finishes


const double deltaN_waterfall_eta_end = -2.0; // End-of-inflation criterion for the waterfall field
const bool deltaN_apply_smoothing = false; // Option to smooth the lattice field before passing to deltaN, can ensure all modes are enough superhorizon
const double deltaN_smoothing_horizon_fraction = 0.1; // Fraction of aH below which to smooth the field before passing to deltaN
const double deltaN_max_fundamental_over_aH = 0.1; // Maximum allowed fundamental mode over aH at handoff


// -------------------- Momentum Cutoff Options --------------------

// Set high_cutoff_index > 0 to cutoff modes with k > (2 pi/L)*high_index and k < (2 pi/L)*low_index
const double high_cutoff_index = 0.0;
const double low_cutoff_index = 0.0;
const double k_uv_cutoff = 0.0; // Disabled when non-positive
const double k_ir_cutoff = 0.0;
const int forcing_cutoff = 0;   // Apply cutoffs only during initialization


// -------------------- Output Configuration --------------------

// Toggle various outputs (1 = enabled)
const int output_spectra = 1;
const int output_histogram = 1;
const int output_energy = 0;
const int output_box3D = 0;
const int output_box2D = 1;
const int output_bispectrum = 0;

const int output_LOG = 0;     // Calculates zeta with the log relation (see documentation)
const double eta_log = -0.5;  // Value of constant eta assumed in the log formula

// Print time/step info to console
const int screen_updates = 1;

// Number of bins used in field histograms
const int nbins = 256;

// Histogram projection mode for multi-field runs:
// 0 = axis-aligned field components only
// 1 = diagonal combinations of the inflaton phi and waterfall field sigma only
// 2 = both axis-aligned and diagonal outputs
const int histogram_projection_mode = 2;
