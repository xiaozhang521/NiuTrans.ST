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
        rootNRe = std::cos(M_2PI / N * forward_sign);
        rootNIm = std::sin(M_2PI / N * forward_sign);
        Real kNRe = -forward_sign, kNIm = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
        // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
        for (INT32 k = 1; 2 * k <= N2; k++) {
            //ComplexMul(rootNRe, rootNIm, &kNRe, &kNIm);

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
            //ComplexAddProduct(Dk_re, Dk_im, kNRe, kNIm, &(data[2 * k]), &(data[2 * k + 1]));

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
                //ComplexAddProduct(Dk_re, -Dk_im, -kNRe, kNIm, &(data[2 * kdash]), &(data[2 * kdash + 1]));
            }
        }

        {  // Now handle k = 0.
          // In simple terms: after the complex fft, data[0] becomes the sum of real
          // parts input[0], input[2]... and data[1] becomes the sum of imaginary
          // pats input[1], input[3]...
          // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
          // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
            Real zeroth = data[0] + data[1],
                n2th = data[0] - data[1];
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
        power_spectrum.SetData(signal_frame, signal_frame->GetDim(0) / 2 + 1, 0);

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

}