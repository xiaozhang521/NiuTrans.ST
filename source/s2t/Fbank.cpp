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
#include "Fbank-function.h"
#include "../niutensor/tensor/core/reduce/ReduceSum.h"
#include "../niutensor/tensor/core/arithmetic/Multiply.h"
#include "../niutensor/tensor/core/math/Unary.h"
using namespace nts;

namespace s2t {

    // After fourier transform transforms a signal from time domain (time domain) to frequency domain (frequency domain), we compute its Power Spectrum.
    // The input "waveform" is 1D?
    void ComputePowerSpectrum(XTensor* waveform) {
        INT32 dim = waveform->GetDim(0);
        INT32 halfDim = dim / 2;

        // handle this special case
        float firstEnergy = waveform->Get1D(0) * waveform->Get1D(0);
        float lastEnergy = waveform->Get1D(1) * waveform->Get1D(1);

        for (INT32 i = 1; i < halfDim; i++) {
            float real = waveform->Get1D(i * 2);
            float im = waveform->Get1D(i * 2 + 1);
            waveform->Set1D(real * real + im * im, i);
        }
        waveform->Set1D(firstEnergy, 0);
        waveform->Set1D(lastEnergy, halfDim); // Will actually never be used, and anyway
        // if the signal has been bandlimited sensibly this should be zero.
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
            RealFft<float>(signal_frame, true);
            
        //RealFft<float>(signal_frame, true);

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