/*
This file contains functions for computing and writing simulation outputs,
such as field means, variances, spectra, and energy densities.
Output files are saved in the 'results/' directory with filenames based on the extension 'ext_'.
*/

#include "main.h"
#include "ffteasy.hpp"
#include <limits>

char name_[550]; // Filenames - set differently by each function to open output files

// Saved during the lattice evolution and used to map k to its horizon-exit time when modes.dat is written after the deltaN endpoint has been found
static std::vector<double> saved_log_a;
static std::vector<double> saved_log_aH;

// Zeroes out a mode and its derivative 
void kill_mode(double *field, double *deriv)
{
    field[0] = 0.;
    field[1] = 0.;
    deriv[0] = 0.;
    deriv[1] = 0.;
    return;
}
inline bool mode_is_cut(int px, int py, int pz)
{
    const double pdisc = sqrt(pw2(px) + pw2(py) + pw2(pz));
    const double kmag = (2.0 * pi / L) * pdisc;
    const double k_fundamental = 2.0 * pi / L;
    const bool physical_cutoff_usable = (k_uv_cutoff <= 0.0 || k_uv_cutoff >= k_fundamental);

    const bool cut_by_index =
        (high_cutoff_index > 0.0) &&
        (pdisc > high_cutoff_index || pdisc < low_cutoff_index);

    const bool cut_by_k =
        (k_uv_cutoff > 0.0) && physical_cutoff_usable &&
        (kmag > k_uv_cutoff || (k_ir_cutoff > 0.0 && kmag < k_ir_cutoff));

    return cut_by_index || cut_by_k;
}

// Computes spatial averages and variances of the field and its derivative
void meansvars(int flush)
{
    static FILE *means_, *vars_, *velocity_;
    DECLARE_INDICES

    float av, var, vel;

    static int first = 1;
    if (first) // Open output files
    {
        snprintf(name_, sizeof(name_), "%s/means%s", path.c_str(), ext_);
        means_ = fopen(name_, mode_);
        snprintf(name_, sizeof(name_), "%s/variance%s", path.c_str(), ext_);
        vars_ = fopen(name_, mode_);
        snprintf(name_, sizeof(name_), "%s/velocity%s", path.c_str(), ext_);
        velocity_ = fopen(name_, mode_);
        first = 0;
    }

    fprintf(means_, "%f", t);
    fprintf(means_, " %e", a);
    fprintf(velocity_, "%f", t);
    fprintf(velocity_, " %e", a);
    fprintf(vars_, "%f", t);
    fprintf(vars_, " %e", a);

    LOOP_MF{
        av = 0.;
        vel = 0.;
        var = 0.;

        LOOP
        {
            av += f[idx_mf(n,i,j,k)];
            vel += fd[idx_mf(n,i,j,k)];
            var += pw2(f[idx_mf(n,i,j,k)]);
        }
        av = av / (float)gridsize; // Convert sum to average
        vel = vel / (float)gridsize;

        vel = vel * pow(a, rescale_s - 1) * rescale_B;

        fprintf(means_, " %.16e", av);
        fprintf(velocity_, " %.16e", vel);
        fprintf(vars_, " %.16e", var - pw2(av));

        if (!std::isfinite(av) || !std::isfinite(vel) || !std::isfinite(var))
        {
            printf("Unstable solution developed. Scalar field not numerical at t=%f\n", t);
            output_parameters();
            fflush(means_);
            fflush(vars_);
            fflush(velocity_);
            exit(1);
        }
}
        fprintf(means_, "\n");
        fprintf(vars_, "\n");
        fprintf(velocity_, "\n");
        if (flush)
        {
            fflush(means_);
            fflush(vars_);
            fflush(velocity_);
        }
}

// Outputs the time and the physical quantities a, adot/a (i.e. Hubble), and adotdot
void scale(int flush)
{
    static FILE *sf_;

    static int first = 1;
    if (first) // Open output file
    {
        snprintf(name_, sizeof(name_), "%s/sf%s", path.c_str(), ext_);
        sf_ = fopen(name_, mode_);
        first = 0;
    }

    const double physical_hubble =
        ad * rescale_B * std::pow(a, rescale_s - 2.0);

    // Output t, a, H, and adotdot in physical units using rescalings
    fprintf(sf_, "%f %f %e %e %e\n",
            t, a,
            physical_hubble,
            pw2(rescale_B) * std::pow(a, 2.0 * rescale_s - 2.0) *
                (ad2 + (rescale_s - 1.0) * pw2(ad) / a),
            log(a));

    const double log_a = std::log(a);
    const double log_aH = std::log(a * physical_hubble);
    if (std::isfinite(log_a) && std::isfinite(log_aH)
        && (saved_log_aH.empty() || log_aH > saved_log_aH.back())) {
        saved_log_a.push_back(log_a);
        saved_log_aH.push_back(log_aH);
    }

    if (flush)
        fflush(sf_);
}


double analytic_potential_2nd_derivativ(const std::vector<double>& field_value, int n) {
    double deriv2 = 0.0;

    if (n == 0){
        deriv2 = Lambda4 * (-2.0 / pw2(mass[1]) + 4.0 * pw2(field_value[1]) / (pw2(phi_c) * pw2(mass[2])));
    }
    if (n == 1) {
        deriv2 = Lambda4 * (-4.0 / pw2(mass[2]) * (1.0 - 3.0 * pw2(field_value[1] / mass[2])) + 4.0 * pw2(field_value[0]) / (pw2(phi_c) * pw2(mass[2])));
    }
    return deriv2 / pw2(rescale_B);
}

// Outputs the effective mass of the waterfall field sigma
void effective_mass(int flush)
{
    static FILE *mass_;
    static int first = 1;
    if (first) // Open output file
    {
        snprintf(name_, sizeof(name_), "%s/mass%s", path.c_str(), ext_);
        mass_ = fopen(name_, mode_);
        first = 0;
    }
    DECLARE_INDICES
    float av;
    std::vector<double> mean_fields(2);

    LOOP_MF{
        av = 0.;

        LOOP
        {
            av += f[idx_mf(n,i,j,k)];
        }
        av = av / (float)gridsize; // Convert sum to average

        mean_fields[n] = av;
    }

    double m2sigma = analytic_potential_2nd_derivativ(mean_fields, 1);
    const double dp = 2. * pi / L;

    const int numbins = (int)(sqrt(3.) * (N / 2)) + 1; // Actual number of bins for the number of dimensions
    const double H_code = ad * std::pow(a, rescale_s - 2.0);
    const double H_physical = rescale_B * H_code;
    const double comoving_hubble_code = a * H_code;

    fprintf(mass_, "# loga a H_physical H_code m2sigma_code bin kcom_code kphys_code omega2_code omega2_over_H2 tachyonic superhorizon mean_waterfall_sigma\n");

    for(int bin= 1; bin < numbins; bin++) {
        double kcom = dp * bin;
        double kphys = kcom * std::pow(a, -1.0);

        double q_k = kcom / comoving_hubble_code;

        double omega2 = m2sigma + pw2(kphys);

        fprintf(mass_, "%e %e %e %e %e %d %e %e %e %e %d %d %e\n",
                log(a), a, H_physical, H_code, m2sigma, bin, kcom,
                kphys, omega2, omega2 / pw2(H_code), (omega2 < 0.0),
                (q_k < 1.0), mean_fields[1]);
    }

    if (flush)
        fflush(mass_);
}

// Outputs power spectrum of the field and applies a high-momentum cutoff if enabled
void spectraf()
{
    static FILE *spectra_, *spectratimes_; // Output files for power spectra and times at which spectra were taken
    const int maxnumbins = (int)(1.73205 * (N / 2)) + 1; // Number of bins (bin spacing=lattice spacing in Fourier space) = sqrt(NDIMS)*(N/2)+1. Set for 3D.
    int numpoints[maxnumbins]; // Number of points in each momentum bin
    float p[maxnumbins], f2[maxnumbins]; // Values for each bin: Momentum, |f_k|^2
    std::vector<std::vector<float>> f2_all(nFields, std::vector<float>(maxnumbins, 0.));
    std::vector<int> numpoints_ref(maxnumbins, 0);
    int numbins = (int)(sqrt(3.) * (N / 2)) + 1; // Actual number of bins for the number of dimensions
    float pmagnitude; // Total momentum (p) in units of lattice spacing, pmagnitude = Sqrt(px^2+py^2+pz^2)
    float dp = 2. * pi / L; // Size of grid spacing in momentum space
    float fp2; // Square magnitude of field (fp2) for a given mode
    int n, i, j, k, px, py, pz, iconj, jconj;
    float norm1 = pow(L / rescale_B, 3) / pow(N, 6); 

    static int first = 1;
    if (first)
    {
        snprintf(name_, sizeof(name_), "%s/spectra%s", path.c_str(), ext_);
        spectra_ = fopen(name_, mode_);

        snprintf(name_, sizeof(name_), "%s/spectratimes%s", path.c_str(), ext_);
        spectratimes_ = fopen(name_, mode_);
        first = 0;
    }

    for (i = 0; i < numbins; i++)
        p[i] = dp * i;

    int arraysize[] = {N, N, N};
    LOOP_MF {

    for (i = 0; i < numbins; i++)
    {
        numpoints[i] = 0;
        f2[i] = 0.;
    }

    double* field_ptr = f.data() + n * N * N * N;
    fftrn(field_ptr, (double*)fnyquist_p[n], 3, arraysize, 1);

    for (i = 0; i < N; i++)
    {
        px = (i <= N / 2 ? i : i - N);
        for (j = 0; j < N; j++)
        {
            py = (j <= N / 2 ? j : j - N);
            for (k = 1; k < N / 2; k++)
            {
                pz = k;
                pmagnitude = sqrt(pw2(px) + pw2(py) + pw2(pz));
                fp2 = pw2(field_ptr[idx(i,j,2 * k)]) + pw2(field_ptr[idx(i,j,2 * k + 1)]);
                numpoints[(int)pmagnitude] += 2;
                f2[(int)pmagnitude] += 2. * fp2;
            }
            for (k = 0; k <= N / 2; k += N / 2)
            {
                pz = k;
                pmagnitude = sqrt(pw2(px) + pw2(py) + pw2(pz));
                if (k == 0)
                {
                    fp2 = pw2(field_ptr[idx(i,j,0)]) + pw2(field_ptr[idx(i,j,1)]);
                }
                else
                {
                    fp2 = pw2(fnyquist_p[n][i][2 * j]) + pw2(fnyquist_p[n][i][2 * j + 1]);
                }
                numpoints[(int)pmagnitude]++;
                f2[(int)pmagnitude] += fp2;
            }
        }
    }

    for (i = 0; i < numbins; i++)
    {
        if (numpoints[i] > 0)
        {
            f2[i] = f2[i] / numpoints[i];
        }
        f2_all[n][i] = f2[i];
        if (n == 0)
            numpoints_ref[i] = numpoints[i];
    }

    if ((high_cutoff_index > 0 || k_uv_cutoff > 0) && forcing_cutoff)
    {
        double* deriv_ptr = fd.data() + n * N * N * N;

        fftrn(deriv_ptr, (double *)fdnyquist_p[n], 3, arraysize, 1);
        for (i = 0; i < N; i++)
        {
            px = (i <= N / 2 ? i : i - N);
            iconj = (i == 0 ? 0 : N - i);
            for (j = 0; j < N; j++)
            {
                py = (j <= N / 2 ? j : j - N);
                for (k = 1; k < N / 2; k++)
                {
                    pz = k;
                    if (mode_is_cut(px, py, pz))
                    {
                        kill_mode(&f[idx_mf(n,i,j,2 * k)], &fd[idx_mf(n,i,j,2 * k)]);
                    }
                }

                if (j > N / 2 || (i > N / 2 && (j == 0 || j == N / 2)))
                {
                    jconj = (j == 0 ? 0 : N - j);

                    if (mode_is_cut(px, py, 0))
                    {
                        kill_mode(&f[idx_mf(n,i,j,0)], &fd[idx_mf(n,i,j,0)]);
                        f[idx_mf(n,iconj,jconj,0)] = f[idx_mf(n,i,j,0)];
                        f[idx_mf(n,iconj,jconj,1)] = -f[idx_mf(n,i,j,1)];
                        fd[idx_mf(n,iconj,jconj,0)] = fd[idx_mf(n,i,j,0)];
                        fd[idx_mf(n,iconj,jconj,1)] = -fd[idx_mf(n,i,j,1)];
                    }

                    if (mode_is_cut(px, py, N / 2))
                    {
                        kill_mode(&fnyquist_p[n][i][2 * j], &fdnyquist_p[n][i][2 * j]);
                        fnyquist_p[n][iconj][2 * jconj] = fnyquist_p[n][i][2 * j];
                        fnyquist_p[n][iconj][2 * jconj + 1] = -fnyquist_p[n][i][2 * j + 1];
                        fdnyquist_p[n][iconj][2 * jconj] = fdnyquist_p[n][i][2 * j];
                        fdnyquist_p[n][iconj][2 * jconj + 1] = -fdnyquist_p[n][i][2 * j + 1];
                    }
                }
                else if ((i == 0 || i == N / 2) && (j == 0 || j == N / 2))
                {
                    if (mode_is_cut(px, py, 0))
                    {
                        kill_mode(&f[idx_mf(n,i,j,0)], &fd[idx_mf(n,i,j,0)]);
                    }

                    if (mode_is_cut(px, py, N / 2))
                    {
                        kill_mode(&fnyquist_p[n][i][2 * j], &fdnyquist_p[n][i][2 * j]);
                    }
                }
            }
        }

        fftrn(deriv_ptr, (double *)fdnyquist_p[n], 3, arraysize, -1);
    }


    fftrn(field_ptr, (double*)fnyquist_p[n], 3, arraysize, -1);
}

    for (i = 0; i < numbins; i++)
    {
        fprintf(spectra_, "%e %d", p[i], numpoints_ref[i]);
        LOOP_MF
            fprintf(spectra_, " %e", norm1 * f2_all[n][i]);
        fprintf(spectra_, "\n");
    }

    fprintf(spectra_, "\n");
    fflush(spectra_);
    fprintf(spectratimes_, "%f %e\n", t, a);
    fflush(spectratimes_);

    return;
}

void cross_spectraf()
{
}

//Outputs the 1D physical momentum, that takes into account the modified dispersion relation (see 2209.13616)
void get_modes()
{
  static FILE *modes_;
  const int maxnumbins=(int)(1.73205*(N/2))+1; // Number of bins (bin spacing=lattice spacing in Fourier space) = sqrt(NDIMS)*(N/2)+1. Set for 3D (i.e. biggest possible).
  int numpoints[maxnumbins]; // Number of points in each momentum bin
  double p_phys[maxnumbins]; // Values for each bin: average physical momentum per bin
  int numbins=(int)(std::sqrt(3.0)*(N/2))+1; // Actual number of bins for the number of dimensions
  double pmagnitude, pphysical; // lattice |p| (bin index) and physical momentum
  int i,j,k,px,py,pz; // px, py, and pz are components of momentum in units of grid spacing
  const double norm1 = rescale_B;
  const double final_efolds = std::log(simulation_initial_a)
      + deltaN_handoff_efolds + deltaN_end_surface_N;

  static int first=1;
  if(first) // Open output files
  {
    snprintf(name_, sizeof(name_), "%s/modes%s", path.c_str(), ext_);
    modes_=fopen(name_,mode_);
    fprintf(modes_, "# k_effective N_k_horizon_exit\n");
    first=0;
  }

  for(i=0;i<numbins;i++) // Initialize all bins to 0
  {
    numpoints[i]=0;
    p_phys[i]=0.;
  }

  for(i=0;i<N;i++)
  {
    px=(i<=N/2 ? i : i-N);
    for(j=0;j<N;j++)
    {
      py=(j<=N/2 ? j : j-N);
      // Modes with 0<k<N/2 are counted twice
      for(k=1;k<N/2;k++)
      {
        pz=k;
        pmagnitude=std::sqrt(pw2((double)px)+pw2((double)py)+pw2((double)pz));
        pphysical=std::sqrt(4.0*pw2((double)N/(double)L)*(pw2(std::sin((double)px*pi/N))+pw2(std::sin((double)py*pi/N))+pw2(std::sin((double)pz*pi/N))));
        numpoints[(int)pmagnitude] += 2;
        p_phys[(int)pmagnitude]    += 2.0*pphysical;
      }
      // k=0 or k=N/2 counted once
      for(k=0;k<=N/2;k+=N/2)
      {
        pz=k;
        pmagnitude=std::sqrt(pw2((double)px)+pw2((double)py)+pw2((double)pz));
        pphysical=std::sqrt(4.0*pw2((double)N/(double)L)*(pw2(std::sin((double)px*pi/N))+pw2(std::sin((double)py*pi/N))+pw2(std::sin((double)pz*pi/N))));

        numpoints[(int)pmagnitude]++; 
        p_phys[(int)pmagnitude] += pphysical;
      }
    }
  }
  for(i=0;i<numbins;i++)
  {
    if(numpoints[i]>0) {
      p_phys[i] = p_phys[i]/numpoints[i];
    }
    const double mode = norm1 * p_phys[i];
    double Nk = std::numeric_limits<double>::quiet_NaN();
    if (mode > 0.0 && saved_log_aH.size() >= 2
        && std::log(mode) >= saved_log_aH.front()
        && std::log(mode) <= saved_log_aH.back()) {
      const double log_mode = std::log(mode);
      size_t upper = 1;
      while (upper < saved_log_aH.size()
             && saved_log_aH[upper] < log_mode) {
        ++upper;
      }
      const size_t lower = upper - 1;
      const double fraction =
          (log_mode - saved_log_aH[lower])
          / (saved_log_aH[upper] - saved_log_aH[lower]);
      const double horizon_exit_efolds = saved_log_a[lower]
          + fraction * (saved_log_a[upper] - saved_log_a[lower]);
      Nk = final_efolds - horizon_exit_efolds;
    }
    fprintf(modes_, "%.17e %.17g\n", mode, Nk);
  }

  fprintf(modes_,"\n");
  fflush(modes_);

  return;
}

void bispectraf()
{
}

void box()
{
    static FILE *box_;
    int i,j,k;
    static int first=1;
    if(first) // Open output files
    {
        snprintf(name_, sizeof(name_), "%s/box%s", path.c_str(), ext_);
        box_ = fopen(name_,mode_);
        first=0;
    }

    for(i=0;i<N;i++) for(j=0;j<N;j++) for(k=0;k<N;k++)
    {
        fprintf(box_,"%.17g\n",f[idx(i,j,k)]);
    }
    fprintf(box_,"\n");
    fflush(box_);
}

void box2d()
{
    static FILE *snapshots_2d_;
    const int i = N / 2;
    int j, k, n;

    static int first=1;
    if(first) // Open output files
    {
        snprintf(name_, sizeof(name_), "%s/snapshots_2di%s", path.c_str(), ext_);
        snapshots_2d_ = fopen(name_,mode_);
        first=0;
    }

    for(j=0;j<N;j++) for(k=0;k<N;k++)
    {
        for (n = 0; n < nFields; ++n)
            fprintf(snapshots_2d_, n == 0 ? "%.17g" : " %.17g",
                    f[idx_mf(n,i,j,k)]);
        fprintf(snapshots_2d_, "\n");
    }
    fprintf(snapshots_2d_,"\n");
    fflush(snapshots_2d_);
}

void box2dot()
{
    static FILE *snapshots_2d_phidot_;
    const int i = N / 2;
    int j, k, n;

    static int first=1;
    if(first) // Open output files
    {
        snprintf(name_, sizeof(name_), "%s/snapshots_2d_phidot%s", path.c_str(), ext_);
        snapshots_2d_phidot_ = fopen(name_,mode_);
        first=0;
    }

    for(j=0;j<N;j++) for(k=0;k<N;k++)
    {
        for (n = 0; n < nFields; ++n)
            fprintf(snapshots_2d_phidot_, n == 0 ? "%.17g" : " %.17g",
                    fd[idx_mf(n,i,j,k)] * rescale_B);
        fprintf(snapshots_2d_phidot_, "\n");
    }
    fprintf(snapshots_2d_phidot_,"\n");
    fflush(snapshots_2d_phidot_);
}

void energy()
{
    static FILE *energy_, *conservation_;
    double deriv_energy, grad_energy, pot_energy;

    double totalE = 0.;
    static int first = 1;
    if (first) // Open output files
    {
        snprintf(name_, sizeof(name_), "%s/energy%s", path.c_str(), ext_);
        energy_ = fopen(name_, mode_);

        snprintf(name_, sizeof(name_), "%s/conservation%s", path.c_str(), ext_);
        conservation_ = fopen(name_, mode_);

        first = 0;
    }

    fprintf(energy_, "%f", t); // Output time
    fprintf(energy_, " %e", a);

    // Calculate and output kinetic (time derivative) energy
    deriv_energy = kin_energy(fd, a);
    totalE += deriv_energy;
    fprintf(energy_, " %e", deriv_energy * rescale_B);

    // Calculate and output gradient energy
    grad_energy = gradient_energy(f, a);
    totalE += grad_energy;
    fprintf(energy_, " %e", grad_energy * rescale_B);

    // Calculate and output potential energy
    pot_energy = potential_energy(f);
    totalE += pot_energy;
    fprintf(energy_, " %e", pot_energy * rescale_B);

    fprintf(energy_, "\n");
    fflush(energy_);

    // Energy conservation
    fprintf(conservation_, "%e %e %e\n",
            t, a, 3.0 * std::pow(a, 2.0 * rescale_s - 4.0) * pw2(ad) / (totalE));
    fflush(conservation_);
}

void readable_time(int t, FILE *info_)
{
    int tminutes = 60, thours = 60 * tminutes, tdays = 24 * thours;

    if (t == 0)
    {
        fprintf(info_, "less than 1 second\n");
        return;
    }

    // Days
    if (t > tdays)
    {
        fprintf(info_, "%d days", t / tdays);
        t = t % tdays;
        if (t > 0)
            fprintf(info_, ", ");
    }
    // Hours
    if (t > thours)
    {
        fprintf(info_, "%d hours", t / thours);
        t = t % thours;
        if (t > 0)
            fprintf(info_, ", ");
    }
    // Minutes
    if (t > tminutes)
    {
        fprintf(info_, "%d minutes", t / tminutes);
        t = t % tminutes;
        if (t > 0)
            fprintf(info_, ", ");
    }
    // Seconds
    if (t > 0)
        fprintf(info_, "%d seconds", t);

    fprintf(info_, "\n");
    return;
}

void histograms()
{
    static FILE *histogram_, *histogramtimes_;
    static std::vector<FILE*> histogram_field_;
    static std::vector<FILE*> histogramtimes_field_;
    static FILE *histogram_diag_plus_, *histogramtimes_diag_plus_;
    static FILE *histogram_diag_minus_, *histogramtimes_diag_minus_;
    // 2D joint histogram files: one per ordered pair (a, b) with a < b
    static std::vector<FILE*> histogram_2d_;
    static std::vector<FILE*> histogramtimes_2d_;
    int i = 0, j = 0, k = 0, n = 0;
    int binnum; // Index of bin for a given field value
    double binfreq[nbins]; // The frequency of field values occurring within each bin
    double bmin, bmax, df; // Minimum and maximum field values for each field and spacing (in field values) between bins
    int numpts; // Count the number of points in the histogram for each field. (Should be all lattice points unless explicit field limits are given.)
    const bool output_axis_histograms =
        histogram_projection_mode == 0 || histogram_projection_mode == 2;
    const bool output_diagonal_histograms =
        (histogram_projection_mode == 1 || histogram_projection_mode == 2) && nFields >= 2;
    const bool output_2d_histograms = nFields >= 2;

    auto write_histogram = [&](FILE *histogram_file, FILE *histogramtimes_file, auto projection) {
        fprintf(histogramtimes_file, "%f", t); // Output time at which histograms were recorded
        fprintf(histogramtimes_file, " %e", a);

        i = 0;
        j = 0;
        k = 0;
        bmin = projection();
        bmax = bmin;
        LOOP
        {
            const double field_value = projection();
            bmin = (field_value < bmin ? field_value : bmin);
            bmax = (field_value > bmax ? field_value : bmax);
        }

        // Find the difference (in field value) between successive bins
        df = (bmax - bmin) / (double)(nbins); // bmin will be at the bottom of the first bin and bmax at the top of the last

        // Initialize all frequencies to zero
        for (i = 0; i < nbins; i++)
            binfreq[i] = 0.;

        // Iterate over grid to determine bin frequencies
        numpts = 0;
        if (df > 0.0)
        {
            LOOP
            {
                const double field_value = projection();
                binnum = (int)((field_value - bmin) / df); // Find index of bin for each value
                if (field_value == bmax) // The maximal field value is at the top of the highest bin
                    binnum = nbins - 1;
                if (binnum >= 0 && binnum < nbins) // Increment frequency in the appropriate bin
                {
                    binfreq[binnum]++;
                    numpts++;
                }
            }
        }
        else
        {
            binfreq[nbins / 2] = (double)gridsize;
            numpts = gridsize;
            df = 1.0;
        }

        for (i = 0; i < nbins; i++)
            fprintf(histogram_file, "%e\n", binfreq[i] / (double)numpts);
        fprintf(histogram_file, "\n");
        fflush(histogram_file);

        fprintf(histogramtimes_file, " %e %e", bmin, df);
        fprintf(histogramtimes_file, "\n");
        fflush(histogramtimes_file);
    };

    // Writes a 2D joint histogram for two field projections.
    // Output: flat nbins*nbins array (row = bin of field_a, col = bin of field_b),
    // normalised so the sum over all cells equals 1.
    // Times file records: t a bmin_a df_a bmin_b df_b
    auto write_histogram_2d = [&](FILE *histogram_file, FILE *histogramtimes_file,
                                  auto proj_a, auto proj_b) {
        fprintf(histogramtimes_file, "%f", t);
        fprintf(histogramtimes_file, " %e", a);

        i = 0; j = 0; k = 0;
        double bmin_a = proj_a(), bmax_a = bmin_a;
        double bmin_b = proj_b(), bmax_b = bmin_b;
        LOOP
        {
            const double va = proj_a();
            const double vb = proj_b();
            bmin_a = (va < bmin_a ? va : bmin_a);
            bmax_a = (va > bmax_a ? va : bmax_a);
            bmin_b = (vb < bmin_b ? vb : bmin_b);
            bmax_b = (vb > bmax_b ? vb : bmax_b);
        }

        const double df_a = (bmax_a > bmin_a) ? (bmax_a - bmin_a) / (double)nbins : 1.0;
        const double df_b = (bmax_b > bmin_b) ? (bmax_b - bmin_b) / (double)nbins : 1.0;

        // Flat 2D frequency array: index [bin_a * nbins + bin_b]
        std::vector<double> freq2d(nbins * nbins, 0.0);
        int numpts2d = 0;

        if (bmax_a > bmin_a && bmax_b > bmin_b)
        {
            LOOP
            {
                const double va = proj_a();
                const double vb = proj_b();
                int ba = (int)((va - bmin_a) / df_a);
                int bb = (int)((vb - bmin_b) / df_b);
                if (va == bmax_a) ba = nbins - 1;
                if (vb == bmax_b) bb = nbins - 1;
                if (ba >= 0 && ba < nbins && bb >= 0 && bb < nbins)
                {
                    freq2d[ba * nbins + bb]++;
                    numpts2d++;
                }
            }
        }
        else
        {
            // All points in the same bin
            freq2d[(nbins / 2) * nbins + (nbins / 2)] = (double)gridsize;
            numpts2d = gridsize;
        }

        const double norm = (numpts2d > 0) ? (double)numpts2d : 1.0;
        for (int idx2d = 0; idx2d < nbins * nbins; ++idx2d)
            fprintf(histogram_file, "%e\n", freq2d[idx2d] / norm);
        fprintf(histogram_file, "\n");
        fflush(histogram_file);

        fprintf(histogramtimes_file, " %e %e %e %e", bmin_a, df_a, bmin_b, df_b);
        fprintf(histogramtimes_file, "\n");
        fflush(histogramtimes_file);
    };

    static int first = 1;
    if (first) // Open output files
    {
        if (output_axis_histograms)
        {
            snprintf(name_, sizeof(name_), "%s/histogram%s", path.c_str(), ext_);
            histogram_ = fopen(name_, mode_);

            snprintf(name_, sizeof(name_), "%s/histogramtimes%s", path.c_str(), ext_);
            histogramtimes_ = fopen(name_, mode_);

            histogram_field_.resize(nFields, nullptr);
            histogramtimes_field_.resize(nFields, nullptr);
            for (int nf = 0; nf < nFields; ++nf)
            {
                snprintf(name_, sizeof(name_), "%s/histogram_field%d%s", path.c_str(), nf, ext_);
                histogram_field_[nf] = fopen(name_, mode_);

                snprintf(name_, sizeof(name_), "%s/histogramtimes_field%d%s", path.c_str(), nf, ext_);
                histogramtimes_field_[nf] = fopen(name_, mode_);
            }
        }

        if (output_diagonal_histograms)
        {
            snprintf(name_, sizeof(name_), "%s/histogram_diag_plus%s", path.c_str(), ext_);
            histogram_diag_plus_ = fopen(name_, mode_);

            snprintf(name_, sizeof(name_), "%s/histogramtimes_diag_plus%s", path.c_str(), ext_);
            histogramtimes_diag_plus_ = fopen(name_, mode_);

            snprintf(name_, sizeof(name_), "%s/histogram_diag_minus%s", path.c_str(), ext_);
            histogram_diag_minus_ = fopen(name_, mode_);

            snprintf(name_, sizeof(name_), "%s/histogramtimes_diag_minus%s", path.c_str(), ext_);
            histogramtimes_diag_minus_ = fopen(name_, mode_);
        }

        if (output_2d_histograms)
        {
            // One file per ordered pair (a, b) with a < b, stored at index a*nFields + b
            const int npairs = nFields * nFields;
            histogram_2d_.resize(npairs, nullptr);
            histogramtimes_2d_.resize(npairs, nullptr);
            for (int na = 0; na < nFields; ++na)
            {
                for (int nb = na + 1; nb < nFields; ++nb)
                {
                    snprintf(name_, sizeof(name_), "%s/histogram_2d_field%d%d%s", path.c_str(), na, nb, ext_);
                    histogram_2d_[na * nFields + nb] = fopen(name_, mode_);

                    snprintf(name_, sizeof(name_), "%s/histogramtimes_2d_field%d%d%s", path.c_str(), na, nb, ext_);
                    histogramtimes_2d_[na * nFields + nb] = fopen(name_, mode_);
                }
            }
        }
        first = 0;
    }

    if (output_axis_histograms)
    {
        for (n = 0; n < nFields; ++n)
        {
            write_histogram(histogram_field_[n], histogramtimes_field_[n], [&]() {
                return f[idx_mf(n, i, j, k)];
            });

            // Preserve the original single-file output for the first field for backward compatibility.
            if (n == 0)
                write_histogram(histogram_, histogramtimes_, [&]() {
                    return f[idx_mf(n, i, j, k)];
                });
        }
    }

    if (output_diagonal_histograms)
    {
        const double diagonal_norm = std::sqrt(2.0);

        write_histogram(histogram_diag_plus_, histogramtimes_diag_plus_, [&]() {
            return (f[idx_mf(0, i, j, k)] + f[idx_mf(1, i, j, k)]) / diagonal_norm;
        });

        write_histogram(histogram_diag_minus_, histogramtimes_diag_minus_, [&]() {
            return (f[idx_mf(0, i, j, k)] - f[idx_mf(1, i, j, k)]) / diagonal_norm;
        });
    }

    if (output_2d_histograms)
    {
        for (int na = 0; na < nFields; ++na)
        {
            for (int nb = na + 1; nb < nFields; ++nb)
            {
                write_histogram_2d(
                    histogram_2d_[na * nFields + nb],
                    histogramtimes_2d_[na * nFields + nb],
                    [&]() { return f[idx_mf(na, i, j, k)]; },
                    [&]() { return f[idx_mf(nb, i, j, k)]; }
                );
            }
        }
    }
}


// -------------------- DeltaN --------------------


static inline void compute_adiabatic_deltaN_log_inputs(std::vector<double>& delta_adiabatic, double& adiabatic_speed)
{
    DECLARE_INDICES

    std::vector<double> mean_field(nFields, 0.0);
    std::vector<double> mean_dphi_dN(nFields, 0.0);

    LOOP_MF {
        LOOP {
            mean_field[n] += f[idx_mf(n, i, j, k)];
            mean_dphi_dN[n] += fd[idx_mf(n, i, j, k)] * pow(a, rescale_s - 1.0) / (ad * pow(a, rescale_s - 2.0));
        }
    }

    for (int n = 0; n < nFields; ++n)
    {
        mean_field[n] /= (double)gridsize;
        mean_dphi_dN[n] /= (double)gridsize;
    }

    adiabatic_speed = 0.0;
    for (int n = 0; n < nFields; ++n)
        adiabatic_speed += pw2(mean_dphi_dN[n]);
    adiabatic_speed = std::sqrt(adiabatic_speed);

    std::vector<double> e_adiabatic(nFields, 0.0);
    if (adiabatic_speed > 0.0)
    {
        for (int n = 0; n < nFields; ++n)
            e_adiabatic[n] = mean_dphi_dN[n] / adiabatic_speed;
    }
    else if (nFields > 0)
    {
        e_adiabatic[0] = 1.0;
    }

    LOOP {
        double adiabatic_perturbation = 0.0;
        for (int n = 0; n < nFields; ++n)
            adiabatic_perturbation += e_adiabatic[n] * (f[idx_mf(n, i, j, k)] - mean_field[n]);
        delta_adiabatic[idx(i, j, k)] = adiabatic_perturbation;
    }
}


void spectraN()
{
    static FILE *spectraN_; // Output file for power spectrum
    const int maxnumbins = (int)(1.73205 * (N / 2)) + 1; // Number of bins (bin spacing=lattice spacing in Fourier space)
    int numpoints[maxnumbins]; // Number of points in each momentum bin
    double p[maxnumbins], f2[maxnumbins]; // Values for each bin: Momentum, |f_k|^2
    int numbins = (int)(sqrt(3.) * (N / 2)) + 1; // Actual number of bins for the number of dimensions
    double pmagnitude; // Total momentum (p) in units of lattice spacing
    // Physical comoving momentum spacing.  This is consistent with the
    // physical-volume normalization used for P_zeta below.
    double dp = rescale_B * 2. * pi / L;
    double fp2; // Square magnitude of field mode
    int i, j, k, px, py, pz;
    double norm1 = pow(L / rescale_B, 3) / pow((double)N, 6);
    int arraysize[] = {N, N, N}; // Array of grid size for FFT routine
    double (*deltaN_nyquist)[2 * N] = fnyquist_p[0];

    static int first = 1;
    if (first) // Open output file
    {
        snprintf(name_, sizeof(name_), "%s/spectraN%s", path.c_str(), ext_);
        spectraN_ = fopen(name_, mode_);
        first = 0;
    }

    // Initialize bin values
    for (i = 0; i < numbins; i++)
    {
        p[i] = dp * i;
        numpoints[i] = 0;
        f2[i] = 0.;
    }

    double Nmean = 0;
    LOOP Nmean += deltaN[idx(i,j,k)];
    Nmean = Nmean / gridsize;
    LOOP deltaN[idx(i,j,k)] -= Nmean;

    // Transform deltaN to Fourier space (double FFT)
    fftrnd(deltaN.data(), (double *)deltaN_nyquist, 3, arraysize, 1);

    // Loop over lattice grid points
    for (i = 0; i < N; i++)
    {
        px = (i <= N / 2 ? i : i - N);
        for (j = 0; j < N; j++)
        {
            py = (j <= N / 2 ? j : j - N);

            // Modes with 0 < k < N/2 are counted twice
            for (k = 1; k < N / 2; k++)
            {
                pz = k;
                pmagnitude = sqrt(pw2((double)px) + pw2((double)py) + pw2((double)pz));
                fp2 = pw2(deltaN[idx(i,j,2 * k)]) + pw2(deltaN[idx(i,j,2 * k + 1)]);
                numpoints[(int)pmagnitude] += 2;
                f2[(int)pmagnitude] += 2. * fp2;
            }

            // Modes with k = 0 or k = N/2 are counted once
            for (k = 0; k <= N / 2; k += N / 2)
            {
                pz = k;
                pmagnitude = sqrt(pw2((double)px) + pw2((double)py) + pw2((double)pz));
                if (k == 0)
                {
                    fp2 = pw2(deltaN[idx(i,j,0)]) + pw2(deltaN[idx(i,j,1)]);
                }
                else
                {
                    fp2 = pw2(deltaN_nyquist[i][2 * j]) + pw2(deltaN_nyquist[i][2 * j + 1]);
                }
                numpoints[(int)pmagnitude]++;
                f2[(int)pmagnitude] += fp2;
            }
        }
    }

    // Output binned power spectrum
    for (i = 0; i < numbins; i++)
    {
        if (numpoints[i] > 0)
        {
            f2[i] = f2[i] / numpoints[i];
        }
        const double Pzeta = norm1 * f2[i];
        const double Delta2zeta = (p[i] > 0.0) ? (pow(p[i], 3.0) / (2.0 * pi * pi)) * Pzeta : 0.0;
        fprintf(spectraN_, "%e %d %e %e", p[i], numpoints[i], Pzeta, Delta2zeta);
        fprintf(spectraN_, "\n");
    }

    // Transform deltaN back to real space
    fftrnd(deltaN.data(), (double *)deltaN_nyquist, 3, arraysize, -1);

    fprintf(spectraN_, "\n");
    fflush(spectraN_);

    return;
}

void boxN()
{
    static FILE *boxN_;
    int i, j, k;
    static int first = 1;
    if (first) // Open output file
    {
        snprintf(name_, sizeof(name_), "%s/boxN%s", path.c_str(), ext_);
        boxN_ = fopen(name_, mode_);
        first = 0;
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
            {
                fprintf(boxN_, "%.17g\n", deltaN[idx(i,j,k)]);
            }
    fprintf(boxN_, "\n");
    fflush(boxN_);
}

void box2dN()
{
    static FILE *snapshots_2d_deltaN_;
    int i = N / 2, j, k;
    static int first = 1;
    if (first) // Open output file
    {
        snprintf(name_, sizeof(name_), "%s/snapshots_2d_deltaN%s", path.c_str(), ext_);
        snapshots_2d_deltaN_ = fopen(name_, mode_);
        first = 0;
    }

    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            fprintf(snapshots_2d_deltaN_, "%.17g\n", deltaN[idx(i,j,k)]);
        }
    fprintf(snapshots_2d_deltaN_, "\n");
    fflush(snapshots_2d_deltaN_);
}

void histogramsN()
{
    static FILE *histogramN_, *histogramtimesN_;
    int i=0, j=0, k=0;
    int binnum;
    double binfreq[nbins];
    double bmin, bmax, df;
    int numpts;

    static int first = 1;
    if (first)
    {
        snprintf(name_, sizeof(name_), "%s/histogramN%s", path.c_str(), ext_);
        histogramN_ = fopen(name_, mode_);

        snprintf(name_, sizeof(name_), "%s/histogramtimesN%s", path.c_str(), ext_);
        histogramtimesN_ = fopen(name_, mode_);
        first = 0;
    }

    fprintf(histogramtimesN_, "%f %e", t, a);

    bmin = deltaN[idx(0,0,0)];
    bmax = bmin;
    LOOP
    {
        bmin = (deltaN[idx(i,j,k)] < bmin ? deltaN[idx(i,j,k)] : bmin);
        bmax = (deltaN[idx(i,j,k)] > bmax ? deltaN[idx(i,j,k)] : bmax);
    }

    df = (bmax - bmin) / (double)(nbins);

    for (i = 0; i < nbins; i++)
        binfreq[i] = 0.;

    numpts = 0;
    if (df > 0.0)
    {
        LOOP
        {
            binnum = (int)((deltaN[idx(i,j,k)] - bmin) / df);
            if (deltaN[idx(i,j,k)] == bmax)
                binnum = nbins - 1;
            if (binnum >= 0 && binnum < nbins)
            {
                binfreq[binnum]++;
                numpts++;
            }
        }
    }
    else
    {
        binfreq[nbins / 2] = (double)gridsize;
        numpts = gridsize;
        df = 1.0;
    }

    for (i = 0; i < nbins; i++)
        fprintf(histogramN_, "%e\n", binfreq[i] / (double)numpts);
    fprintf(histogramN_, "\n");
    fflush(histogramN_);

    fprintf(histogramtimesN_, " %e %e\n", bmin, df);
    fflush(histogramtimesN_);
}

void spectraLOG()
{
//     static FILE *spectraLOG_; // Output files for power spectra and times at which spectra were taken
//     const int maxnumbins = (int)(1.73205 * (N / 2)) + 1;
//     int numpoints[maxnumbins];
//     double p[maxnumbins], f2[maxnumbins];
//     int numbins = (int)(sqrt(3.) * (N / 2)) + 1;
//     double pmagnitude;
//     double dp = 2. * pi / L;
//     double fp2;
//     int i, j, k, px, py, pz;
//     double norm1 = pow(L / rescale_B, 3) / pow((double)N, 6);
//     int arraysize[] = {N, N, N};

//     static int first = 1;
//     if (first)
//     {
//         snprintf(name_, sizeof(name_), "%s/spectraLOG%s", path.c_str(), ext_);
//         spectraLOG_ = fopen(name_, mode_);
//         first = 0;
//     }

//     for (i = 0; i < numbins; i++)
//         p[i] = dp * i;

//     for (i = 0; i < numbins; i++)
//     {
//         numpoints[i] = 0;
//         f2[i] = 0.;
//     }

//     std::vector<double> delta_adiabatic(gridsize, 0.0);
//     double factt = 0.0;
//     compute_adiabatic_deltaN_log_inputs(delta_adiabatic, factt);

//     LOOP
//     {
//         deltaN[idx(i,j,k)] = 0;
//         if (factt > 0.0 && (1 - eta_log * delta_adiabatic[idx(i,j,k)] / factt) > 0)
//             deltaN[idx(i,j,k)] = 1./eta_log * log(1 - eta_log * delta_adiabatic[idx(i,j,k)] / factt);
//     }

//     fftrnd(deltaN.data(), (double *)fnyquist_p[n], 3, arraysize, 1);

//     for (i = 0; i < N; i++)
//     {
//         px = (i <= N / 2 ? i : i - N);
//         for (j = 0; j < N; j++)
//         {
//             py = (j <= N / 2 ? j : j - N);
//             for (k = 1; k < N / 2; k++)
//             {
//                 pz = k;
//                 pmagnitude = sqrt(pw2((double)px) + pw2((double)py) + pw2((double)pz));
//                 fp2 = pw2(deltaN[idx(i,j,2 * k)]) + pw2(deltaN[idx(i,j,2 * k + 1)]);
//                 numpoints[(int)pmagnitude] += 2;
//                 f2[(int)pmagnitude] += 2. * fp2;
//             }
//             for (k = 0; k <= N / 2; k += N / 2)
//             {
//                 pz = k;
//                 pmagnitude = sqrt(pw2((double)px) + pw2((double)py) + pw2((double)pz));
//                 if (k == 0)
//                 {
//                     fp2 = pw2(deltaN[idx(i,j,0)]) + pw2(deltaN[idx(i,j,1)]);
//                 }
//                 else
//                 {
//                     fp2 = pw2(fnyquist_p[n][i][2 * j]) + pw2(fnyquist_p[n][i][2 * j + 1]);
//                 }
//                 numpoints[(int)pmagnitude]++;
//                 f2[(int)pmagnitude] += fp2;
//             }
//         }
//     }

//     for (i = 0; i < numbins; i++)
//     {
//         if (numpoints[i] > 0)
//         {
//             f2[i] = f2[i] / numpoints[i];
//         }
//         fprintf(spectraLOG_, "%e %d %e\n", p[i], numpoints[i], norm1 * f2[i]);
//     }

//     fftrnd(deltaN.data(), (double *)fnyquist_p[n], 3, arraysize, -1);

//     fprintf(spectraLOG_, "\n");
//     fflush(spectraLOG_);

//     return;
}

void histogramsLOG()
{
    static FILE *histogramLOG_, *histogramtimesLOG_;
    int i=0, j=0, k=0;
    int binnum;
    double binfreq[nbins];
    double bmin, bmax, df;
    int numpts;

    static int first = 1;
    if (first)
    {
        snprintf(name_, sizeof(name_), "%s/histogramLOG%s", path.c_str(), ext_);
        histogramLOG_ = fopen(name_, mode_);

        snprintf(name_, sizeof(name_), "%s/histogramtimesLOG%s", path.c_str(), ext_);
        histogramtimesLOG_ = fopen(name_, mode_);
        first = 0;
    }

    fprintf(histogramtimesLOG_, "%f %e", t, a);

    // The log approximation is evaluated by save_last(), before initializeN()
    // normally allocates the deltaN scratch field.
    deltaN.assign(gridsize, 0.0);
    std::vector<double> delta_adiabatic(gridsize, 0.0);
    double factt = 0.0;
    compute_adiabatic_deltaN_log_inputs(delta_adiabatic, factt);

    LOOP
    {
        deltaN[idx(i,j,k)] = 0;
        if (factt > 0.0 && (1 - eta_log * delta_adiabatic[idx(i,j,k)] / factt) > 0)
            deltaN[idx(i,j,k)] = 1./eta_log * log(1 - eta_log * delta_adiabatic[idx(i,j,k)] / factt);
    }

    bmin = deltaN[idx(0,0,0)];
    bmax = bmin;
    LOOP
    {
        if (factt > 0.0 && 1 - eta_log * delta_adiabatic[idx(i,j,k)] / factt > 0)
        {
            bmin = (deltaN[idx(i,j,k)] < bmin ? deltaN[idx(i,j,k)] : bmin);
            bmax = (deltaN[idx(i,j,k)] > bmax ? deltaN[idx(i,j,k)] : bmax);
        }
    }

    df = (bmax - bmin) / (double)(nbins);

    for (i = 0; i < nbins; i++)
        binfreq[i] = 0.;

    numpts = 0;
    LOOP
    {
        if (factt > 0.0 && 1 - eta_log * delta_adiabatic[idx(i,j,k)] / factt > 0)
        {
            binnum = (int)((deltaN[idx(i,j,k)] - bmin) / df);
            if (deltaN[idx(i,j,k)] == bmax)
                binnum = nbins - 1;
            if (binnum >= 0 && binnum < nbins)
            {
                binfreq[binnum]++;
                numpts++;
            }
        }
    }

    for (i = 0; i < nbins; i++)
        fprintf(histogramLOG_, "%e\n", binfreq[i] / (double)numpts);
    fprintf(histogramLOG_, "\n");
    fflush(histogramLOG_);

    fprintf(histogramtimesLOG_, " %e %e\n", bmin, df);
    fflush(histogramtimesLOG_);
}


// Output information about the run parameters.
// This need only be called at the beginning and end of the run.
void output_parameters()
{
    int n;
    static FILE *info_;
    static time_t tStart, tFinish; // Keep track of elapsed clock time

    static int first = 1;
    if (first) // At beginning of run output run parameters
    {
        snprintf(name_, sizeof(name_), "%s/info%s", path.c_str(), ext_);
        info_ = fopen(name_, mode_);

        fprintf(info_, "--------------------------\n");
        fprintf(info_, "General Program Information\n");
        fprintf(info_, "-----------------------------\n");
        fprintf(info_, "Grid size=%d^%d\n", N, 3);
        fprintf(info_, "L=%.17g\n\n", L);
        LOOP_MF{
        fprintf(info_, "Mu_%d=%.17g\n", (n+1), mass[n]);
        }
        fprintf(info_, "Coupling parameter Lambda4=%.17g\n", Lambda4);
        fprintf(info_, "Mass M=%.17g\n", mass[2]);
        fprintf(info_, "Critical phi=%.17g\n", phi_c);

        fprintf(info_, "rescale_s=%f\n", rescale_s);
        fprintf(info_, "rescale_B=%e\n", rescale_B);
        
        time(&tStart);
        fprintf(info_, "\nRun began at %s", ctime(&tStart)); // Output date in readable form
        first = 0;
    }
    else // If not at beginning record elapsed time for run
    {
        time(&tFinish);
        fprintf(info_, "Run ended at %s", ctime(&tFinish)); // Output ending date
        fprintf(info_, "\nRun from t=%f to t=%f took ", t0, t);
        readable_time((int)(tFinish - tStart), info_);
        output_efold_summary(info_);
        fprintf(info_, "\n");
    }

    fflush(info_);
    return;
}

void output_efold_summary(FILE* info_)
{
    const double handoff_efolds = std::isfinite(deltaN_handoff_efolds)
        ? deltaN_handoff_efolds
        : log(a / simulation_initial_a);
    const double deltaN_efolds = deltaN_end_surface_N;
    const double total_efolds = handoff_efolds + deltaN_efolds;

    fprintf(info_, "\n-------------------------\n");
    fprintf(info_, "Simulation e-fold summary\n");
    fprintf(info_, "-------------------------\n");
    fprintf(info_, "Total number of e-folds=%.17g\n", total_efolds);
    if (std::isfinite(critical_crossing_efolds)) {
        fprintf(info_, "Number of e-folds to critical point=%.17g\n",
                critical_crossing_efolds);
        fprintf(info_, "Length of waterfall=%.17g\n",
                total_efolds - critical_crossing_efolds);
        fprintf(info_, "Number of e-folds from critical point to deltaN handoff=%.17g\n",
                handoff_efolds - critical_crossing_efolds);
    } else {
        fprintf(info_, "Number of e-folds to critical point=not reached\n");
        fprintf(info_, "Length of waterfall=not available\n");
        fprintf(info_, "Number of e-folds from critical point to deltaN handoff=not available\n");
    }
    fprintf(info_, "Number of e-folds in deltaN calculation=%.17g\n",
            deltaN_efolds);
}

// Calculate and save quantities (means, variances, etc.). If force>0 all infrequent calculations will be performed
void save(int infrequent)
{

    meansvars(infrequent);
    effective_mass(infrequent);
    scale(infrequent);

    // Infrequent calculations
    if (infrequent)
    {
        if (output_box3D)
            box();
        if (output_box2D)
        {
            box2d();
            box2dot();
        }
        if (output_energy)
            energy();
        if (output_spectra)
        {
            spectraf();
        }
        if (output_histogram)
            histograms();
    }
}

void save_last()
{
    if (output_bispectrum)
        bispectraf();

    if (output_LOG)
    {
        //careful if you place these functions somewhere else, they use the vector deltaN as ausiliary variable
        spectraLOG();
        histogramsLOG();
    }
}

void saveN()
{
    double Nmean = 0.0;
    int i, j, k;

    LOOP
    {
        Nmean += deltaN[idx(i,j,k)];
    }
    Nmean /= static_cast<double>(gridsize);

    LOOP
    {
        deltaN[idx(i,j,k)] -= Nmean;
    }

    histogramsN();

    if (output_box2D)
        box2dN();

    if (output_spectra)
        spectraN();
}
