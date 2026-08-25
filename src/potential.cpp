// potential.cpp — Inflationary potential 
//
// This file defines the inflationary potential and its derivatives.
// For analytic models (e.g. m²ϕ²), customize `analytic_potential` and its derivative.
// For numerical models, potential values are interpolated from tabulated data.

#include "main.h"

// -------------------------------------------------------------
// Analytic potential (default: Hybrid inflation)
// -------------------------------------------------------------

static inline double analytic_potential_values(double phi, double sigma) {
    const double potential_value = Lambda4 * (
        pw2(1.0 - pw2(sigma / mass[2]))
        + (phi - phi_c) / mass[0]
        - pw2(phi - phi_c) / pw2(mass[1])
        + 2.0 * pw2(phi) * pw2(sigma) / (pw2(phi_c) * pw2(mass[2])));
    return potential_value / pw2(rescale_B);
}
static inline double analytic_potential_derivative_values(
    double phi, double sigma, int n) {
    double derivative = 0.0;
    if (n == 0) {
        derivative = Lambda4 * (
            1.0 / mass[0]
            - 2.0 * (phi - phi_c) / pw2(mass[1])
            + 4.0 * phi * pw2(sigma) / (pw2(phi_c) * pw2(mass[2])));
    } else if (n == 1 && nFields > 1) {
        derivative = Lambda4 * (
            -4.0 * sigma / pw2(mass[2]) * (1.0 - pw2(sigma / mass[2]))
            + 4.0 * pw2(phi) * sigma / (pw2(phi_c) * pw2(mass[2])));
    }
    return derivative / pw2(rescale_B);
}

double analytic_potential(const std::vector<double>& field_value) {
    // Hybrid inflation potential 
    // V = Lambda4 ( (1 - (sigma/M)^2)^2 +(phi - phi_c) / mu1 - (phi-phi_c)^2/mu2^2 + 2 * phi^2 * sigma^2 / (phi_c^2 * M^2) )
    
    return analytic_potential_values(field_value[0], field_value[1]);
}

double analytic_potential_derivative(const std::vector<double>& field_value, int n) {
    return analytic_potential_derivative_values(
        field_value[0], field_value[1], n);
}

double analytic_potential_2nd_derivative(const std::vector<double>& field_value, int n) {
    double deriv2 = 0.0;

    if (n == 0){
        // d²V/dφ² = Lambda4 ( -2 / mu2^2 + 4 * sigma^2 / (phi_c^2 * M^2) )
        deriv2 = Lambda4 * (-2.0 / pw2(mass[1]) + 4.0 * pw2(field_value[1]) / (pw2(phi_c) * pw2(mass[2])));
    }
    if (n == 1) {
        // d²V/dσ² = Lambda4 ( -4/M^2 * (1 - 3*(sigma/M)^2) + 4 * phi^2 / (phi_c^2 * M^2) )
        deriv2 = Lambda4 * (-4.0 / pw2(mass[2]) * (1.0 - 3.0 * pw2(field_value[1] / mass[2])) + 4.0 * pw2(field_value[0]) / (pw2(phi_c) * pw2(mass[2])));
    }
    return deriv2 / pw2(rescale_B);
}

// -------------------------------------------------------------
// Slow-roll parameters
// -------------------------------------------------------------
double epsilon(const std::vector<double>& field_value, int n) {
    double V = potential(field_value);

    double dV = analytic_potential_derivative(field_value, n);

    return 0.5 * (dV*dV) / (V*V);
}

double eta(const std::vector<double>& field_value, int n) {
    double V = potential(field_value);
    double d2V = analytic_potential_2nd_derivative(field_value, n);
    return d2V / V;
}


// -------------------------------------------------------------
// Return the potential V(ϕ) at a given field value (interpolated or analytic)
// -------------------------------------------------------------
double potential(const std::vector<double>& field_value) {
    return analytic_potential(field_value);
}

double potential_derivative_at_state(const std::vector<double>& field_value, int n) {
    return analytic_potential_derivative(field_value, n);
}

// -------------------------------------------------------------
// Return ∂V/∂ϕ at a grid point (interpolated or analytic)
// -------------------------------------------------------------
double potential_derivative(double field_value, int n, int i, int j, int k, const std::vector<double>* field_state) {
    const std::vector<double>& state = field_state ? *field_state : f;
    double phi = state[idx_mf(0,i,j,k)];
    double sigma = state[idx_mf(1,i,j,k)];
    if (!field_state) {
        if (n == 0) phi = field_value;
        if (n == 1) sigma = field_value;
    }
    return analytic_potential_derivative_values(phi, sigma, n);
}

// -------------------------------------------------------------
// Compute the total potential energy on the grid
// -------------------------------------------------------------
double potential_energy(const std::vector<double>& field) {
    int i, j, k;
    double pot = 0.0;

#if parallel_calculation
#pragma omp parallel for reduction(+:pot) collapse(3)
#endif
    LOOP {
        pot += analytic_potential_values(
            field[idx_mf(0,i,j,k)], field[idx_mf(1,i,j,k)]);
    }

    pot /= static_cast<double>(gridsize);
    return pot;
}

// -------------------------------------------------------------
// Compute ∂V/∂ϕ divided by V at a grid point
// Used in δN evolution
// -------------------------------------------------------------
double pot_ratio(int n, int i, int j, int k) {
    std::vector<double> field_values(nFields);
    for (int m = 0; m < nFields; m++) {
        field_values[m] = f[idx_mf(m,i,j,k)];
    }
    double pot = potential(field_values);
    double pot_deriv = potential_derivative(f[idx_mf(n,i,j,k)], n, i, j, k);

    return pot_deriv / pot;
}
