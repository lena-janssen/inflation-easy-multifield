// main.cpp - Program entry point and top-level driver for InflationEasy

#include "main.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

// Field values and derivatives
std::vector<double> f;
std::vector<double> fd;
std::vector<double> deltaN;

double fnyquist_p[nFields][N][2*N], fdnyquist_p[nFields][N][2*N];

double t, t0;
double astep = 1.0, a = 1.0, ad = 0.0, ad2 = 0.0, ad0 = 0.0, Ne = 0.0;
double deltaN_end_surface_N = 0.0;
double hubble_init = 0.0;
double simulation_initial_a = 1.0;
double critical_crossing_efolds = NAN;
double deltaN_handoff_efolds = NAN;
std::vector<double> initial_field_derivative_used(nFields, 0.0);
std::vector<double> initial_field_efold_derivative_used(nFields, 0.0);
char mode_[10] = "w";
char ext_[500] = ".dat";

#if perform_deltaN
std::vector<double> phiref(nFields, 0.0); // Reference field values for deltaN calculation

#endif


// Ensure the 'results' directory exists
bool ensure_results_directory() {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    if (mkdir(path.c_str(), 0777) != 0) {
      std::cerr << "Failed to create results directory.\n";
      return false;
    }
  } else if (!(info.st_mode & S_IFDIR)) {
    std::cerr << "Path exists but is not a directory.\n";
    return false;
  }
  return true;
}

int main() {
  if (seed < 1) {
    printf("Warning: The parameter seed has been set to %d, which will result in incorrect output.\n", seed);
  }

  if (!ensure_results_directory()) {
    return 1;
  }

  FILE* output_ = fopen((path + "/output.txt").c_str(), "w");
  if (!output_) {
    std::cerr << "Failed to open output file.\n";
    return 1;
  }

  initialize_simulation();     // Initialize the simulation
  double phi_mean = 0.0, phi_var = 0.0;
  double sigma_mean = 0.0, sigma_var = 0.0;

  int i,j,k;
  LOOP {
      double phi = f[idx_mf(0,i,j,k)];
      double sigma = f[idx_mf(1,i,j,k)];

      phi_mean += phi;
      sigma_mean += sigma;
      phi_var += phi*phi;
      sigma_var += sigma*sigma;
  }

  double Vgrid = double(N*N*N);

  phi_mean /= Vgrid;
  sigma_mean /= Vgrid;
  phi_var = phi_var/Vgrid - phi_mean*phi_mean;
  sigma_var = sigma_var/Vgrid - sigma_mean*sigma_mean;

  printf("phi_mean - phi_c = %.16e\n", phi_mean - phi_c);
  printf("phi_rms fluct    = %.16e\n", sqrt(phi_var));
  printf("sigma_mean       = %.16e\n", sigma_mean);
  printf("sigma_rms fluct  = %.16e\n", sqrt(sigma_var));
  run_evolution_loop(output_); // Run the main evolution loop
  deltaN_handoff_efolds = log(a / simulation_initial_a);

#if perform_deltaN
  run_deltaN_loop(output_);    // Run the deltaN evolution loop
#endif

  // The deltaN endpoint is now known, so modes.dat can include N_k measured
  // backward from the common final surface.
  get_modes();


  output_parameters();
  fprintf(output_, "InflationEasy program finished\n");
  printf("InflationEasy program finished\n");

  fclose(output_);
  return 0;
}
