/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
  *
  * $Created by: HE Erfeng (heerfeng1023@gmail.com) 2023-09
  */
#include "../niutensor/tensor/XTensor.h"
#include "../niuTensor/tensor/core/math/Clip.h"
#include <stdint.h>
#include "Fbank.h"
#include "utils.h"
#include "../niutensor/tensor/core/reduce/ReduceSum.h"
#include "../niutensor/tensor/core/arithmetic/Multiply.h"
#include "../niutensor/tensor/core/math/Unary.h"
using namespace nts;

namespace s2t {

    template<typename Real>
    SplitRadixComplexFft<Real>::SplitRadixComplexFft(INT32 N) {
        if ((N & (N - 1)) != 0 || N <= 1)
            ASSERT(FALSE);
        N_ = N;
        logn_ = 0;
        while (N > 1) {
            N >>= 1;
            logn_++;
        }
        ComputeTables();
    }

    template<typename Real>
    SplitRadixComplexFft<Real>::~SplitRadixComplexFft() {
        delete[] brseed_;
        if (tab_ != NULL) {
            for (INT32 i = 0; i < logn_ - 3; i++)
                delete[] tab_[i];
            delete[] tab_;
        }
    }

    template <typename Real>
    SplitRadixComplexFft<Real>::SplitRadixComplexFft(
        const SplitRadixComplexFft<Real>& other) :
        N_(other.N_), logn_(other.logn_) {
        // This code duplicates tables from a previously computed object.
        // Compare with the code in ComputeTables().
        INT32 lg2 = logn_ >> 1;
        if (logn_ & 1) lg2++;
        INT32 brseed_size = 1 << lg2;
        brseed_ = new INT32[brseed_size];
        std::memcpy(brseed_, other.brseed_, sizeof(INT32) * brseed_size);

        if (logn_ < 4) {
            tab_ = NULL;
        }
        else {
            tab_ = new Real * [logn_ - 3];
            for (INT32 i = logn_; i >= 4; i--) {
                INT32 m = 1 << i, m2 = m / 2, m4 = m2 / 2;
                INT32 this_array_size = 6 * (m4 - 2);
                tab_[i - 4] = new Real[this_array_size];
                std::memcpy(tab_[i - 4], other.tab_[i - 4],
                    sizeof(Real) * this_array_size);
            }
        }
    }

    // After fourier transform transforms a signal from time domain (time domain) to frequency domain (frequency domain), we compute its Power Spectrum.
    // The input "waveform" is 1D?
    void ComputePowerSpectrum(XTensor waveform) {
        INT32 dim = waveform.GetDim(0);
        INT32 halfDim = dim / 2;

        // handle this special case
        float firstEnergy = waveform.Get1D(0) * waveform.Get1D(0); 
        float lastEnergy = waveform.Get1D(1) * waveform.Get1D(1);

        for (INT32 i = 1; i < halfDim; i++) {
            float real = waveform.Get1D(i * 2);
            float im = waveform.Get1D(i * 2 + 1);
            waveform.Set1D(real * real + im * im, i);
        }
        waveform.Set1D(firstEnergy, 0);
        waveform.Set1D(lastEnergy, halfDim); // Will actually never be used, and anyway
        // if the signal has been bandlimited sensibly this should be zero.
    }

    template<typename Real>
    void SplitRadixComplexFft<Real>::ComputeRecursive(Real* xr, Real* xi, INT32 logn) const {

        INT32    m, m2, m4, m8, nel, n;
        Real* xr1, * xr2, * xi1, * xi2;
        Real* cn = nullptr, * spcn = nullptr, * smcn = nullptr, * c3n = nullptr,
            * spc3n = nullptr, * smc3n = nullptr;
        Real    tmp1, tmp2;
        Real   sqhalf = M_SQRT1_2;

        /* Check range of logn */
        if (logn < 0)
            ASSERT(FALSE);
            //KALDI_ERR << "Error: logn is out of bounds in SRFFT";

        /* Compute trivial cases */
        if (logn < 3) {
            if (logn == 2) {  /* length m = 4 */
                xr2 = xr + 2;
                xi2 = xi + 2;
                tmp1 = *xr + *xr2;
                *xr2 = *xr - *xr2;
                *xr = tmp1;
                tmp1 = *xi + *xi2;
                *xi2 = *xi - *xi2;
                *xi = tmp1;
                xr1 = xr + 1;
                xi1 = xi + 1;
                xr2++;
                xi2++;
                tmp1 = *xr1 + *xr2;
                *xr2 = *xr1 - *xr2;
                *xr1 = tmp1;
                tmp1 = *xi1 + *xi2;
                *xi2 = *xi1 - *xi2;
                *xi1 = tmp1;
                xr2 = xr + 1;
                xi2 = xi + 1;
                tmp1 = *xr + *xr2;
                *xr2 = *xr - *xr2;
                *xr = tmp1;
                tmp1 = *xi + *xi2;
                *xi2 = *xi - *xi2;
                *xi = tmp1;
                xr1 = xr + 2;
                xi1 = xi + 2;
                xr2 = xr + 3;
                xi2 = xi + 3;
                tmp1 = *xr1 + *xi2;
                tmp2 = *xi1 + *xr2;
                *xi1 = *xi1 - *xr2;
                *xr2 = *xr1 - *xi2;
                *xr1 = tmp1;
                *xi2 = tmp2;
                return;
            }
            else if (logn == 1) {   /* length m = 2 */
                xr2 = xr + 1;
                xi2 = xi + 1;
                tmp1 = *xr + *xr2;
                *xr2 = *xr - *xr2;
                *xr = tmp1;
                tmp1 = *xi + *xi2;
                *xi2 = *xi - *xi2;
                *xi = tmp1;
                return;
            }
            else if (logn == 0) return;   /* length m = 1 */
        }

        /* Compute a few constants */
        m = 1 << logn; m2 = m / 2; m4 = m2 / 2; m8 = m4 / 2;


        /* Step 1 */
        xr1 = xr; xr2 = xr1 + m2;
        xi1 = xi; xi2 = xi1 + m2;
        for (n = 0; n < m2; n++) {
            tmp1 = *xr1 + *xr2;
            *xr2 = *xr1 - *xr2;
            xr2++;
            *xr1++ = tmp1;
            tmp2 = *xi1 + *xi2;
            *xi2 = *xi1 - *xi2;
            xi2++;
            *xi1++ = tmp2;
        }

        /* Step 2 */
        xr1 = xr + m2; xr2 = xr1 + m4;
        xi1 = xi + m2; xi2 = xi1 + m4;
        for (n = 0; n < m4; n++) {
            tmp1 = *xr1 + *xi2;
            tmp2 = *xi1 + *xr2;
            *xi1 = *xi1 - *xr2;
            xi1++;
            *xr2++ = *xr1 - *xi2;
            *xr1++ = tmp1;
            *xi2++ = tmp2;
            // xr1++; xr2++; xi1++; xi2++;
        }

        /* Steps 3 & 4 */
        xr1 = xr + m2; xr2 = xr1 + m4;
        xi1 = xi + m2; xi2 = xi1 + m4;
        if (logn >= 4) {
            nel = m4 - 2;
            cn = tab_[logn - 4]; spcn = cn + nel;  smcn = spcn + nel;
            c3n = smcn + nel;  spc3n = c3n + nel; smc3n = spc3n + nel;
        }
        xr1++; xr2++; xi1++; xi2++;
        // xr1++; xi1++;
        for (n = 1; n < m4; n++) {
            if (n == m8) {
                tmp1 = sqhalf * (*xr1 + *xi1);
                *xi1 = sqhalf * (*xi1 - *xr1);
                *xr1 = tmp1;
                tmp2 = sqhalf * (*xi2 - *xr2);
                *xi2 = -sqhalf * (*xr2 + *xi2);
                *xr2 = tmp2;
            }
            else {
                tmp2 = *cn++ * (*xr1 + *xi1);
                tmp1 = *spcn++ * *xr1 + tmp2;
                *xr1 = *smcn++ * *xi1 + tmp2;
                *xi1 = tmp1;
                tmp2 = *c3n++ * (*xr2 + *xi2);
                tmp1 = *spc3n++ * *xr2 + tmp2;
                *xr2 = *smc3n++ * *xi2 + tmp2;
                *xi2 = tmp1;
            }
            xr1++; xr2++; xi1++; xi2++;
        }

        /* Call ssrec again with half DFT length */
        ComputeRecursive(xr, xi, logn - 1);

        /* Call ssrec again twice with one quarter DFT length.
           Constants have to be recomputed, because they are static! */
           // m = 1 << logn; m2 = m / 2;
        ComputeRecursive(xr + m2, xi + m2, logn - 2);
        // m = 1 << logn;
        m4 = 3 * (m / 4);
        ComputeRecursive(xr + m4, xi + m4, logn - 2);
    }

    template<typename Real>
    void SplitRadixComplexFft<Real>::BitReversePermute(Real* x, INT32 logn) const {
        INT32      i, j, lg2, n;
        INT32      off, fj, gno, * brp;
        Real    tmp, * xp, * xq;

        lg2 = logn >> 1;
        n = 1 << lg2;
        if (logn & 1) lg2++;

        /* Unshuffling loop */
        for (off = 1; off < n; off++) {
            fj = n * brseed_[off]; i = off; j = fj;
            tmp = x[i]; x[i] = x[j]; x[j] = tmp;
            xp = &x[i];
            brp = &(brseed_[1]);
            for (gno = 1; gno < brseed_[off]; gno++) {
                xp += n;
                j = fj + *brp++;
                xq = x + j;
                tmp = *xp; *xp = *xq; *xq = tmp;
            }
        }
    }

    template<typename Real>
    void SplitRadixComplexFft<Real>::Compute(Real* xr, Real* xi, bool forward) const {
        if (!forward) {  // reverse real and imaginary parts for complex FFT.
            Real* tmp = xr;
            xr = xi;
            xi = tmp;
        }
        ComputeRecursive(xr, xi, logn_);
        if (logn_ > 1) {
            BitReversePermute(xr, logn_);
            BitReversePermute(xi, logn_);
        }
    }

    template<typename Real>
    void SplitRadixComplexFft<Real>::Compute(Real* x, bool forward,
        std::vector<Real>* temp_buffer) const {
        ASSERT(temp_buffer != NULL);
        if (temp_buffer->size() != N_)
            temp_buffer->resize(N_);
        Real* temp_ptr = &((*temp_buffer)[0]);
        for (INT32 i = 0; i < N_; i++) {
            x[i] = x[i * 2];  // put the real part in the first half of x.
            temp_ptr[i] = x[i * 2 + 1];  // put the imaginary part in temp_buffer.
        }
        // copy the imaginary part back to the second half of x.
        memcpy(static_cast<void*>(x + N_),
            static_cast<void*>(temp_ptr),
            sizeof(Real) * N_);

        Compute(x, x + N_, forward);
        // Now change the format back to interleaved.
        memcpy(static_cast<void*>(temp_ptr),
            static_cast<void*>(x + N_),
            sizeof(Real) * N_);
        for (INT32 i = N_ - 1; i > 0; i--) {  // don't include 0,
            // in case INT32 is unsigned, the loop would not terminate.
            // Treat it as a special case.
            x[i * 2] = x[i];
            x[i * 2 + 1] = temp_ptr[i];
        }
        x[1] = temp_ptr[0];  // special case of i = 0.
    }

    template<typename Real>
    void SplitRadixRealFft<Real>::Compute(Real* data, bool forward) {
        Compute(data, forward, &this->temp_buffer_);
    }

    template<typename Real>
    void SplitRadixRealFft<Real>::Compute(Real* data, bool forward, std::vector<Real>* temp_buffer) const {
        INT32 N = N_, N2 = N / 2;
        ASSERT(N % 2 == 0);
        if (forward) // call to base class
            SplitRadixComplexFft<Real>::Compute(data, true, temp_buffer);

        Real rootNRe, rootNIm;  // exp(-2pi/N), forward; exp(2pi/N), backward
        int forward_sign = forward ? -1 : 1;
        rootNRe = cos(M_2PI / N * forward_sign);
        rootNIm = sin(M_2PI / N * forward_sign);
        Real kNRe = -forward_sign, kNIm = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
        // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
        for (INT32 k = 1; 2 * k <= N2; k++) {

            //Complex multiplication
            Real tmpRe = (kNRe * rootNRe) - (kNIm * rootNIm);
            kNIm = (kNRe * rootNRe) + (kNIm * rootNIm);
            kNRe = tmpRe;

            Real Ck_re, Ck_im, Dk_re, Dk_im;
            // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
            Ck_re = 0.5 * (data[2 * k] + data[N - 2 * k]);
            Ck_im = 0.5 * (data[2 * k + 1] - data[N - 2 * k + 1]);
            // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
            Dk_re = 0.5 * (data[2 * k + 1] + data[N - 2 * k + 1]);
            // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
            Dk_im = -0.5 * (data[2 * k] - data[N - 2 * k]);
            // A_k = C_k + 1^(k/N) D_k:
            data[2 * k] = Ck_re;  // A_k <-- C_k
            data[2 * k + 1] = Ck_im;
            // now A_k += D_k 1^(k/N)
            ComplexAddProduct(Dk_re, Dk_im, kNRe, kNIm, &(data[2 * k]), &(data[2 * k + 1]));

            INT32 kdash = N2 - k;
            if (kdash != k) {
                // Next we handle the index k' = N/2 - k.  This is necessary
                // to do now, to avoid invalidating data that we will later need.
                // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
                // and D_k, so the equations are simple modifications of the above,
                // replacing Ck_im and Dk_im with their negatives.
                data[2 * kdash] = Ck_re;  // A_k' <-- C_k'
                data[2 * kdash + 1] = -Ck_im;
                // now A_k' += D_k' 1^(k'/N)
                // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 * (1^(k/N))^*
                // so it's the same as 1^(k/N) but with the real part negated.
                ComplexAddProduct(Dk_re, -Dk_im, -kNRe, kNIm, &(data[2 * kdash]), &(data[2 * kdash + 1]));
            }
        }

        {  // Now handle k = 0.
          // In simple terms: after the complex fft, data[0] becomes the sum of real
          // parts input[0], input[2]... and data[1] becomes the sum of imaginary
          // pats input[1], input[3]...
          // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
          // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
            Real zeroth = data[0] + data[1], n2th = data[0] - data[1];
            data[0] = zeroth;
            data[1] = n2th;
            if (!forward) {
                data[0] /= 2;
                data[1] /= 2;
            }
        }
        
        if (!forward) {  // call to base class
            SplitRadixComplexFft<Real>::Compute(data, false, temp_buffer);
            for (INT32 i = 0; i < N; i++)
                data[i] *= 2.0;
            // This is so we get a factor of N increase, rather than N/2 which we would
            // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in dimension N/2.
            // It's for consistency with our normal FFT convensions.
        }
        
    }

    template<typename Real>
    void SplitRadixComplexFft<Real>::ComputeTables() {
        INT32    imax, lg2, i, j;
        INT32     m, m2, m4, m8, nel, n;
        Real* cn, * spcn, * smcn, * c3n, * spc3n, * smc3n;
        Real    ang, c, s;

        lg2 = logn_ >> 1;
        if (logn_ & 1) lg2++;
        brseed_ = new INT32[1 << lg2];
        brseed_[0] = 0;
        brseed_[1] = 1;
        for (j = 2; j <= lg2; j++) {
            imax = 1 << (j - 1);
            for (i = 0; i < imax; i++) {
                brseed_[i] <<= 1;
                brseed_[i + imax] = brseed_[i] + 1;
            }
        }

        if (logn_ < 4) {
            tab_ = NULL;
        }
        else {
            tab_ = new Real * [logn_ - 3];
            for (i = logn_; i >= 4; i--) {
                /* Compute a few constants */
                m = 1 << i; m2 = m / 2; m4 = m2 / 2; m8 = m4 / 2;

                /* Allocate memory for tables */
                nel = m4 - 2;

                tab_[i - 4] = new Real[6 * nel];

                /* Initialize pointers */
                cn = tab_[i - 4]; spcn = cn + nel;  smcn = spcn + nel;
                c3n = smcn + nel;  spc3n = c3n + nel; smc3n = spc3n + nel;

                /* Compute tables */
                for (n = 1; n < m4; n++) {
                    if (n == m8) continue;
                    ang = n * M_2PI / m;
                    c = std::cos(ang); s = std::sin(ang);
                    *cn++ = c; *spcn++ = -(s + c); *smcn++ = s - c;
                    ang = 3 * n * M_2PI / m;
                    c = std::cos(ang); s = std::sin(ang);
                    *c3n++ = c; *spc3n++ = -(s + c); *smc3n++ = s - c;
                }
            }
        }
    }

    FbankComputer::FbankComputer(const FbankOptions& opts) :
        opts_(opts), srfft_(NULL) {
        if (opts.energy_floor > 0.0)
            log_energy_floor_ = logf(opts.energy_floor);

        INT32 padded_window_size = opts.frame_opts.PaddedWindowSize();
        if ((padded_window_size & (padded_window_size - 1)) == 0)  // Is a power of two.
            srfft_ = new SplitRadixRealFft<float>(padded_window_size);

        // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
        // [note: this call caches it.]
        GetMelBanks(1.0);
    }

    FbankComputer::FbankComputer(const FbankComputer& other) :
        opts_(other.opts_), log_energy_floor_(other.log_energy_floor_),
        mel_banks_(other.mel_banks_), srfft_(NULL) {
        for (std::map<float, MelBanks*>::iterator iter = mel_banks_.begin();
            iter != mel_banks_.end();
            ++iter)
            iter->second = new MelBanks(*(iter->second));
        if (other.srfft_)
            srfft_ = new SplitRadixRealFft<float>(*(other.srfft_));
    }

    FbankComputer::~FbankComputer() {
        for (std::map<float, MelBanks*>::iterator iter = mel_banks_.begin();
            iter != mel_banks_.end(); ++iter)
            delete iter->second;
        delete srfft_;
    }

    const MelBanks* FbankComputer::GetMelBanks(float vtln_warp) {
        MelBanks* this_mel_banks = NULL;
        std::map<float, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
        if (iter == mel_banks_.end()) {
            this_mel_banks = new MelBanks(opts_.mel_opts,
                opts_.frame_opts,
                vtln_warp);
            mel_banks_[vtln_warp] = this_mel_banks;
        }
        else {
            this_mel_banks = iter->second;
        }
        return this_mel_banks;
    }

    // Compute Fbank, I recommend read the code from here.
    void FbankComputer::Compute(float signal_raw_log_energy,
        float vtln_warp,
        XTensor* signal_frame,
        XTensor* feature) {

        const MelBanks& mel_banks = *(GetMelBanks(vtln_warp));

        ASSERT(signal_frame->GetDim(0) == opts_.frame_opts.PaddedWindowSize() && feature->GetDim(0) == this->Dim());


        // Compute energy after window function (not the raw one).
        if (opts_.use_energy && !opts_.raw_energy)
            signal_raw_log_energy = logf(std::max<float>(ReduceSum(Multiply(signal_frame, signal_frame, 0), 0).Get0D(), std::numeric_limits<float>::epsilon()));

        if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
        {
            int startIndex = { 0 };
            srfft_->Compute(reinterpret_cast<float*>(signal_frame->GetCell(&startIndex, 1)), true); // Need a pointer to the start of the vector's data.
        }
        else  // An alternative algorithm that works for non-powers-of-two.
            ASSERT(FALSE);

        // Convert the FFT into a power spectrum.
        ComputePowerSpectrum(signal_frame);
        XTensor power_spectrum(signal_frame);
        int dimSize = { signal_frame->GetDim(0) / 2 + 1 };
        power_spectrum.SetDim(&dimSize);
        int index = { 0 };
        power_spectrum.SetData(signal_frame->GetCell(&index, 1), signal_frame->GetDim(0) / 2 + 1, 0);

        // I commented out this code because I haven't found an existing way to root every element of XTensor yet, 
        // and we don't use magnitude for the time being
        // Use magnitude instead of power if requested.
        //if (!opts_.use_power)
        //    power_spectrum.ApplyPow(0.5);

        INT32 mel_offset = ((opts_.use_energy && !opts_.htk_compat) ? 1 : 0);
        XTensor mel_energies(*feature);
        dimSize = { opts_.mel_opts.num_bins };
        mel_energies.SetDim(&dimSize);
        mel_energies.SetData(feature, opts_.mel_opts.num_bins, mel_offset);

        // Sum with mel fiterbanks over the power spectrum
        mel_banks.Compute(power_spectrum, &mel_energies);
        if (opts_.use_log_fbank) {
            // Avoid log of zero (which should be prevented anyway by dithering).
            ClipMe(mel_energies, std::numeric_limits<float>::epsilon(), FLT_MAX);
            LogMe(mel_energies);
        }

        // Copy energy as first value (or the last, if htk_compat == true).
        if (opts_.use_energy) {
            if (opts_.energy_floor > 0.0 && signal_raw_log_energy < log_energy_floor_) {
                signal_raw_log_energy = log_energy_floor_;
            }
            INT32 energy_index = opts_.htk_compat ? opts_.mel_opts.num_bins : 0;
            feature->Set1D(signal_raw_log_energy, energy_index);
        }
    }

    // ------------MelBanks VtlnWarpFreq------------
    float MelBanks::VtlnWarpFreq(float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
        float vtln_high_cutoff,
        float low_freq,  // upper+lower frequency cutoffs in mel computation
        float high_freq,
        float vtln_warp_factor,
        float freq) {
        /// This computes a VTLN warping function that is not the same as HTK's one,
        /// but has similar inputs (this function has the advantage of never producing
        /// empty bins).

        /// This function computes a warp function F(freq), defined between low_freq and
        /// high_freq inclusive, with the following properties:
        ///  F(low_freq) == low_freq
        ///  F(high_freq) == high_freq
        /// The function is continuous and piecewise linear with two inflection
        ///   points.
        /// The lower inflection point (measured in terms of the unwarped
        ///  frequency) is at frequency l, determined as described below.
        /// The higher inflection point is at a frequency h, determined as
        ///   described below.
        /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
        /// If the higher inflection point (measured in terms of the unwarped
        ///   frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
        ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
        ///   max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
        ///   h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
        ///     = vtln_high_cutoff * min(1, vtln_warp_factor).
        /// If the lower inflection point (measured in terms of the unwarped
        ///   frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
        ///   This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
        ///                       = vtln_low_cutoff * max(1, vtln_warp_factor)


        if (freq < low_freq || freq > high_freq) return freq;  // in case this gets called
        // for out-of-range frequencies, just return the freq.

        ASSERT(vtln_low_cutoff > low_freq &&
            "be sure to set the --vtln-low option higher than --low-freq");
        ASSERT(vtln_high_cutoff < high_freq &&
            "be sure to set the --vtln-high option lower than --high-freq [or negative]");
        float one = 1.0;
        float l = vtln_low_cutoff * std::max<float>(one, vtln_warp_factor);
        float h = vtln_high_cutoff * std::min<float>(one, vtln_warp_factor);
        float scale = 1.0 / vtln_warp_factor;
        float Fl = scale * l;  // F(l);
        float Fh = scale * h;  // F(h);
        ASSERT(l > low_freq && h < high_freq);
        // slope of left part of the 3-piece linear function
        float scale_left = (Fl - low_freq) / (l - low_freq);
        // [slope of center part is just "scale"]

        // slope of right part of the 3-piece linear function
        float scale_right = (high_freq - Fh) / (high_freq - h);

        if (freq < l) {
            return low_freq + scale_left * (freq - low_freq);
        }
        else if (freq < h) {
            return scale * freq;
        }
        else {  // freq >= h
            return high_freq + scale_right * (freq - high_freq);
        }
    }

    float MelBanks::VtlnWarpMelFreq(float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
        float vtln_high_cutoff,
        float low_freq,  // upper+lower frequency cutoffs in mel computation
        float high_freq,
        float vtln_warp_factor,
        float mel_freq) {
        return MelScale(VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
            low_freq, high_freq,
            vtln_warp_factor, InverseMelScale(mel_freq)));
    }

    MelBanks::MelBanks(const MelBanksOptions& opts,
        const FrameExtractionOptions& frame_opts,
        float vtln_warp_factor) :
        htk_mode_(opts.htk_mode) {
        INT32 num_bins = opts.num_bins;
        if (num_bins < 3) ASSERT(FALSE); // Must have at least 3 mel bins
        float sample_freq = frame_opts.samp_freq;
        INT32 window_length_padded = frame_opts.PaddedWindowSize();
        ASSERT(window_length_padded % 2 == 0);
        INT32 num_fft_bins = window_length_padded / 2;
        float nyquist = 0.5 * sample_freq;

        float low_freq = opts.low_freq, high_freq;
        if (opts.high_freq > 0.0)
            high_freq = opts.high_freq;
        else
            high_freq = nyquist + opts.high_freq;

        // low-freq should lower than high-freq, and both low-freq and high-freq shoud be higher than 0.0. 
        // Besides, nyquist should be the highest among the three.
        if (low_freq < 0.0 || low_freq >= nyquist
            || high_freq <= 0.0 || high_freq > nyquist
            || high_freq <= low_freq)
            ASSERT(FALSE); 
        
        float fft_bin_width = sample_freq / window_length_padded;
        // fft-bin width [think of it as Nyquist-freq / half-window-length]

        float mel_low_freq = MelScale(low_freq);
        float mel_high_freq = MelScale(high_freq);

        debug_ = opts.debug_mel;

        // divide by num_bins+1 in next line because of end-effects where the bins
        // spread out to the sides.
        float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);

        float vtln_low = opts.vtln_low,
            vtln_high = opts.vtln_high;
        if (vtln_high < 0.0) {
            vtln_high += nyquist;
        }

        if (vtln_warp_factor != 1.0 &&
            (vtln_low < 0.0 || vtln_low <= low_freq
                || vtln_low >= high_freq
                || vtln_high <= 0.0 || vtln_high >= high_freq
                || vtln_high <= vtln_low))
            ASSERT(FALSE);
           /* KALDI_ERR << "Bad values in options: vtln-low " << vtln_low
            << " and vtln-high " << vtln_high << ", versus "
            << "low-freq " << low_freq << " and high-freq "
            << high_freq;*/

        bins_.resize(num_bins);
        int dimSize = { num_bins };
        center_freqs_.Resize(1, &dimSize);

        for (INT32 bin = 0; bin < num_bins; bin++) {
            float left_mel = mel_low_freq + bin * mel_freq_delta,
                center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
                right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

            if (vtln_warp_factor != 1.0) {
                left_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                    vtln_warp_factor, left_mel);
                center_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                    vtln_warp_factor, center_mel);
                right_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                    vtln_warp_factor, right_mel);
            }
            center_freqs_.Set1D(InverseMelScale(center_mel), bin);
            // this_bin will be a vector of coefficients that is only
            // nonzero where this mel bin is active.
            XTensor thisBin;
            dimSize = { num_fft_bins };
            thisBin.Resize(1, &dimSize);
            INT32 first_index = -1, last_index = -1;
            for (INT32 i = 0; i < num_fft_bins; i++) {
                float freq = (fft_bin_width * i);  // Center frequency of this fft
                // bin.
                float mel = MelScale(freq);
                if (mel > left_mel && mel < right_mel) {
                    float weight;
                    if (mel <= center_mel)
                        weight = (mel - left_mel) / (center_mel - left_mel);
                    else
                        weight = (right_mel - mel) / (right_mel - center_mel);
                    thisBin.Set1D(weight, i);
                    if (first_index == -1)
                        first_index = i;
                    last_index = i;
                }
            }
            ASSERT(first_index != -1 && last_index >= first_index
                && "You may have set --num-mel-bins too large.");

            bins_[bin].first = first_index;
            INT32 size = last_index + 1 - first_index;
            dimSize = { size };
            bins_[bin].second.Resize(1, &dimSize);
            int index = { first_index };
            bins_[bin].second.SetData(thisBin.GetCell(&index, 1), size , 0);

            // Replicate a bug in HTK, for testing purposes.
            if (opts.htk_mode && bin == 0 && mel_low_freq != 0.0)
                bins_[bin].second.Set1D(0.0, 0);
        }
    }

    MelBanks::MelBanks(const MelBanks& other) :
        center_freqs_(other.center_freqs_),
        bins_(other.bins_),
        debug_(other.debug_),
        htk_mode_(other.htk_mode_) { }

    void MelBanks::Compute(const XTensor& power_spectrum,
        XTensor* mel_energies_out) const {
        INT32 num_bins = bins_.size();
        ASSERT(mel_energies_out->GetDim(0) == num_bins);

        for (INT32 i = 0; i < num_bins; i++) {
            INT32 offset = bins_[i].first;
            const XTensor& v(bins_[i].second);
            XTensor partPowerSpectrum(&power_spectrum);
            partPowerSpectrum.Reshape(v.GetDim(0));
            partPowerSpectrum.SetData(power_spectrum.GetCell(&offset, 1), v.GetDim(0), 0);
            float energy = ReduceSum(Multiply(v, partPowerSpectrum, 0), 0).Get0D();
            // HTK-like flooring- for testing purposes (we prefer dither)
            if (htk_mode_ && energy < 1.0) energy = 1.0;
            mel_energies_out->Set1D(energy, i);

            // The following assert was added due to a problem with OpenBlas that
            // we had at one point (it was a bug in that library).  Just to detect
            // it early.
            ASSERT(!std::isnan(mel_energies_out->Get1D(i)));
        }

    }
}