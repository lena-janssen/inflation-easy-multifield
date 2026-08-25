// main.h - Global declarations and shared definitions for InflationEasy
//
// This header declares global variables, constants, and functions
// used across the simulation. Global variable definitions are in main.cpp.

#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <fstream>
#include <vector>
#include "parameters.h" // Adjustable simulation parameters


// -------------f------- Math and Constants --------------------

#define float double // Work with double precision 

const double pi = (double)(2. * asin(1.));

inline double pw2(double x) { return x * x; } // Square of a number


// ----------------- Function to access vector elements -----------------

inline size_t idx(int i, int j, int k) {
    return static_cast<size_t>(i) * N * N + static_cast<size_t>(j) * N + static_cast<size_t>(k);
}

inline size_t idx_mf(int n, int i, int j, int k) {
    return static_cast<size_t>(n) * N * N * N + static_cast<size_t>(i) * N * N + static_cast<size_t>(j) * N + static_cast<size_t>(k);
}


// -------------------- Simulation Parameters --------------------

// Grid spacing (comoving distance between points)
const double dx = L / (double)N;

// Total number of points in the 3D grid
const int gridsize = N * N * N;

// Total number of points in the 4D grid (inclduing multiple fields)
const int gridsize_mf = nFields * N * N * N;


// -------------------- Global Dynamic Variables --------------------

// Time and scale factor evolution
extern double t, t0;             // Time and initial time
extern double astep, a;          // Scale factor and backup (for leapfrog)
extern double ad, ad2;           // First and second derivatives of scale factor
extern double aterm;             // Intermediate quantity used in evolution
extern double Ne;                // e-folding number used in deltaN evolution
extern double hubble_init;       // Initial Hubble parameter
extern double simulation_initial_a;
extern double critical_crossing_efolds;
extern double deltaN_handoff_efolds;
extern std::vector<double> initial_field_derivative_used;
extern std::vector<double> initial_field_efold_derivative_used;
extern double deltaN_end_surface_N;

// Reference field value(s) for deltaN termination
#if perform_deltaN
extern std::vector<double> phiref;
#endif

// I/O formatting
extern char ext_[500];          // Filename extension for outputs
extern char mode_[];            // File open mode (write/append)


// -------------------- Lattice Fields --------------------

// Main field and its time derivative (double)
extern std::vector<double> f;
extern std::vector<double> fd;

// deltaN field for separate evolution (double)
extern std::vector<double> deltaN;

// Nyquist frequency components for FFT
extern double fnyquist_p[nFields][N][2 * N], fdnyquist_p[nFields][N][2 * N];


// -------------------- Grid Macros --------------------
                             
#define LOOP for(i=0;i<N;i++) for(j=0;j<N;j++) for(k=0;k<N;k++) // Loop over full grid
#define INDEXLIST int i, int j, int k                           // Function parameter list for indexing
#define DECLARE_INDICES int n, i, j, k;                         // Local index variable declarations
#define LOOP_MF for(n=0;n<nFields;n++)                          // Loop over fields


// -------------------- Function Declarations --------------------

// Initialization
void initialize();               // Basic parameter checks and hubble_init
void initializef();              // Generate vacuum fluctuations
void initialize_simulation();    // Full initialization pipeline
void initializeN();              // Setup for deltaN calculation

// Field evolution
void evolve_fields(double d);     // Evolve f with step d
void evolve_derivs(double d);     // Evolve fd with step d
void evolve_scale(double d);      // (Unused but present)
void evolve_fieldsN(double d);    // Evolve deltaN with step d
void evolve_derivsN(double d);    // Evolve deltaN derivative with step d

// Energies
float gradient_energy(const std::vector<double>& field, double a);         
double kin_energy(const std::vector<double>& velocity, double scale_factor);
float potential_energy(const std::vector<double>& field);      

// Potential interface
double potential(const std::vector<double>& field_value);   
double potential_derivative_at_state(const std::vector<double>& field_value, int n);
double potential_derivative(double field_value, int n, int i, int j, int k, const std::vector<double>* field_state = nullptr); 
double pot_ratio(int n, int i, int j, int k);     
double epsilon(const std::vector<double>& field_values, int n);
double eta(const std::vector<double>& field_values, int n);

void smooth_deltaN_field(std::vector<double>& field, double smoothing_scale);
void smooth_all_fields_for_deltaN();

// Output routines
void output_parameters();               
void output_efold_summary(FILE* info_);
void get_modes();
void save(int force);                   
void save_last();                     
void saveN();                          
// Utilities
bool ensure_results_directory();                                              

// Main evolution loops
void run_evolution_loop(FILE* output_);
void run_deltaN_loop(FILE* output_);
