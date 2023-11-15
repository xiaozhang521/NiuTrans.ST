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
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
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
        srfft_(NULL) {
        opts_ = opts;
        
        
        if (opts_.melOpts.customFilter != "") {
            filter.order = 2;
            int filterDimSize[] = { opts_.melOpts.numBins , opts_.frameOpts.WindowSize() / 2 + 1 };
            filter.Resize(2, filterDimSize);
            std::ifstream inputFile(opts_.melOpts.customFilter);
            ASSERT(inputFile.is_open());
            std::vector<std::vector<std::string>> csvData;
            std::string line;

            while (std::getline(inputFile, line)) {
                std::vector<std::string> row;
                std::istringstream lineStream(line);
                std::string cell;

                while (std::getline(lineStream, cell, ',')) {
                    row.push_back(cell);
                }

                csvData.push_back(row);
            }

            inputFile.close();
            int rowCount = 0;

            for (const auto& row : csvData) {
                int column = 0;

                for (const std::string& cell : row) {

                    try {
                        float value = std::stod(cell);
                        filter.Set2D(value, rowCount, column);
                    }
                    catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid number format: " << cell << std::endl;
                    }
                    column++;

                }
                rowCount++;
            }

        }
        
        if (opts.energyFloor > 0.0)
            logEnergyFloor_ = logf(opts.energyFloor);

        INT32 padded_window_size = opts.frameOpts.PaddedWindowSize();
        if ((padded_window_size & (padded_window_size - 1)) == 0)  // Is a power of two.
            srfft_ = new SplitRadixRealFft<float>(padded_window_size);

        // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
        // [note: this call caches it.]
        GetMelBanks(1.0);
    }

    FbankComputer::FbankComputer(const FbankComputer& other) :
        opts_(other.opts_), logEnergyFloor_(other.logEnergyFloor_),
        mel_banks_(other.mel_banks_), srfft_(NULL) {
        opts_ = other.opts_;
        filter = other.filter;
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
            this_mel_banks = new MelBanks(opts_.melOpts,
                opts_.frameOpts,
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
        XTensor &feature) {

        const MelBanks& mel_banks = *(GetMelBanks(vtln_warp));

        ASSERT(signal_frame->GetDim(0) == opts_.frameOpts.PaddedWindowSize());


        // Compute energy after window function (not the raw one).
        if (opts_.useEnergy && !opts_.rawEnergy)
            signal_raw_log_energy = logf(std::max<float>(ReduceSum(Multiply(signal_frame, signal_frame, 0), 0).Get0D(), std::numeric_limits<float>::epsilon()));

        if (opts_.oneSide) {
            OneSidedFFT<float>(signal_frame, true);

            int startIndex = { 0 };
            std::vector<float> myOneSideFFTIn(reinterpret_cast<float*>(signal_frame->GetCell(&startIndex, 1)),
                reinterpret_cast<float*>(signal_frame->GetCell(&startIndex, 1)) + signal_frame->GetDim(0));
            std::vector<std::complex<float>> mymyOneSideFFTOut = OneSidedFFT(myOneSideFFTIn);

        }
        else {
            if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
            {
                int startIndex = { 0 };
                srfft_->Compute(reinterpret_cast<float*>(signal_frame->GetCell(&startIndex, 1)), true); // Need a pointer to the start of the vector's data.
            }
            else  // An alternative algorithm that works for non-powers-of-two.
                RealFft<float>(signal_frame, true);
        }

        // Convert the FFT into a power spectrum.
        ComputePowerSpectrum(signal_frame);
        XTensor power_spectrum;
        power_spectrum.order = 2;
        int dimSize[] = {signal_frame->GetDim(0) / 2 + 1 , 1};
        power_spectrum.Resize(2, dimSize);
        int index = { 0 };
        power_spectrum.SetData(signal_frame->GetCell(&index, 1), signal_frame->GetDim(0) / 2 + 1, 0);

        // We don't use magnitude for the time being
        // Use magnitude instead of power if requested.
        if (!opts_.usePower)
            SqrtMe(power_spectrum);

        INT32 mel_offset = ((opts_.useEnergy && !opts_.htkCompat) ? 1 : 0);
        XTensor mel_energies(feature);
        // It's a tensor with two dim.
        int melDimSize = { opts_.melOpts.numBins };
        mel_energies.SetDim(&melDimSize);
        int startIndex = { 0 };
        mel_energies.SetData(feature.GetCell(&startIndex, 1), opts_.melOpts.numBins, mel_offset);

        // Sum with mel fiterbanks over the power spectrum
        if (opts_.melOpts.customFilter != ""){
            mel_energies = MMul(filter, X_NOTRANS, power_spectrum, X_NOTRANS);
        }
        else{
            mel_banks.Compute(power_spectrum, &mel_energies);
        }
        
        if (opts_.useLogFbank) {
            
            // Avoid log of zero (which should be prevented anyway by dithering).
            ClipMe(mel_energies, 1e-10, FLT_MAX);
            Log10Me(mel_energies);
        }

        // Copy energy as first value (or the last, if htkCompat == true).
        if (opts_.useEnergy) {
            if (opts_.energyFloor > 0.0 && signal_raw_log_energy < logEnergyFloor_) {
                signal_raw_log_energy = logEnergyFloor_;
            }
            INT32 energy_index = opts_.htkCompat ? opts_.melOpts.numBins : 0;
            feature.Set1D(signal_raw_log_energy, energy_index);
        }

        feature = mel_energies;
    }

    // ------------MelBanks VtlnWarpFreq------------
    float MelBanks::VtlnWarpFreq(float vtlnLow_cutoff,  // upper+lower frequency cutoffs for VTLN.
        float vtlnHigh_cutoff,
        float lowFreq,  // upper+lower frequency cutoffs in mel computation
        float highFreq,
        float vtln_warp_factor,
        float freq) {
        /// This computes a VTLN warping function that is not the same as HTK's one,
        /// but has similar inputs (this function has the advantage of never producing
        /// empty bins).

        /// This function computes a warp function F(freq), defined between lowFreq and
        /// highFreq inclusive, with the following properties:
        ///  F(lowFreq) == lowFreq
        ///  F(highFreq) == highFreq
        /// The function is continuous and piecewise linear with two inflection
        ///   points.
        /// The lower inflection point (measured in terms of the unwarped
        ///  frequency) is at frequency l, determined as described below.
        /// The higher inflection point is at a frequency h, determined as
        ///   described below.
        /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
        /// If the higher inflection point (measured in terms of the unwarped
        ///   frequency) is at h, then max(h, F(h)) == vtlnHigh_cutoff.
        ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
        ///   max(h, h/vtln_warp_factor) == vtlnHigh_cutoff, so
        ///   h = vtlnHigh_cutoff / max(1, 1/vtln_warp_factor).
        ///     = vtlnHigh_cutoff * min(1, vtln_warp_factor).
        /// If the lower inflection point (measured in terms of the unwarped
        ///   frequency) is at l, then min(l, F(l)) == vtlnLow_cutoff
        ///   This implies that l = vtlnLow_cutoff / min(1, 1/vtln_warp_factor)
        ///                       = vtlnLow_cutoff * max(1, vtln_warp_factor)


        if (freq < lowFreq || freq > highFreq) return freq;  // in case this gets called
        // for out-of-range frequencies, just return the freq.

        ASSERT(vtlnLow_cutoff > lowFreq &&
            "be sure to set the --vtln-low option higher than --low-freq");
        ASSERT(vtlnHigh_cutoff < highFreq &&
            "be sure to set the --vtln-high option lower than --high-freq [or negative]");
        float one = 1.0;
        float l = vtlnLow_cutoff * std::max<float>(one, vtln_warp_factor);
        float h = vtlnHigh_cutoff * std::min<float>(one, vtln_warp_factor);
        float scale = 1.0 / vtln_warp_factor;
        float Fl = scale * l;  // F(l);
        float Fh = scale * h;  // F(h);
        ASSERT(l > lowFreq && h < highFreq);
        // slope of left part of the 3-piece linear function
        float scale_left = (Fl - lowFreq) / (l - lowFreq);
        // [slope of center part is just "scale"]

        // slope of right part of the 3-piece linear function
        float scale_right = (highFreq - Fh) / (highFreq - h);

        if (freq < l) {
            return lowFreq + scale_left * (freq - lowFreq);
        }
        else if (freq < h) {
            return scale * freq;
        }
        else {  // freq >= h
            return highFreq + scale_right * (freq - highFreq);
        }
    }

    float MelBanks::VtlnWarpMelFreq(float vtlnLow_cutoff,  // upper+lower frequency cutoffs for VTLN.
        float vtlnHigh_cutoff,
        float lowFreq,  // upper+lower frequency cutoffs in mel computation
        float highFreq,
        float vtln_warp_factor,
        float mel_freq) {
        return MelScale(VtlnWarpFreq(vtlnLow_cutoff, vtlnHigh_cutoff,
            lowFreq, highFreq,
            vtln_warp_factor, InverseMelScale(mel_freq)));
    }

    MelBanks::MelBanks(const MelBanksOptions& opts,
        const FrameExtractionOptions& frameOpts,
        float vtln_warp_factor) :
        htkMode_(opts.htkMode) {
        INT32 numBins = opts.numBins;
        if (numBins < 3) ASSERT(FALSE); // Must have at least 3 mel bins
        float sample_freq = frameOpts.sampFreq;
        INT32 window_length_padded = frameOpts.PaddedWindowSize();
        ASSERT(window_length_padded % 2 == 0);
        INT32 num_fft_bins = window_length_padded / 2;
        float nyquist = 0.5 * sample_freq;

        float lowFreq = opts.lowFreq, highFreq;
        if (opts.highFreq > 0.0)
            highFreq = opts.highFreq;
        else
            highFreq = nyquist + opts.highFreq;

        // low-freq should lower than high-freq, and both low-freq and high-freq shoud be higher than 0.0. 
        // Besides, nyquist should be the highest among the three.
        if (lowFreq < 0.0 || lowFreq >= nyquist
            || highFreq <= 0.0 || highFreq > nyquist
            || highFreq <= lowFreq)
            ASSERT(FALSE); 
        
        float fft_bin_width = sample_freq / window_length_padded;
        // fft-bin width [think of it as Nyquist-freq / half-window-length]

        float mel_lowFreq = MelScale(lowFreq);
        float mel_highFreq = MelScale(highFreq);

        debug_ = opts.debugMel;

        // divide by numBins+1 in next line because of end-effects where the bins
        // spread out to the sides.
        float mel_freq_delta = (mel_highFreq - mel_lowFreq) / (numBins + 1);

        float vtlnLow = opts.vtlnLow,
            vtlnHigh = opts.vtlnHigh;
        if (vtlnHigh < 0.0) {
            vtlnHigh += nyquist;
        }

        if (vtln_warp_factor != 1.0 &&
            (vtlnLow < 0.0 || vtlnLow <= lowFreq
                || vtlnLow >= highFreq
                || vtlnHigh <= 0.0 || vtlnHigh >= highFreq
                || vtlnHigh <= vtlnLow))
            ASSERT(FALSE);
           /* ERR << "Bad values in options: vtln-low " << vtlnLow
            << " and vtln-high " << vtlnHigh << ", versus "
            << "low-freq " << lowFreq << " and high-freq "
            << highFreq;*/

        bins_.resize(numBins);
        int dimSize = { numBins };
        center_freqs_.Resize(1, &dimSize);

        for (INT32 bin = 0; bin < numBins; bin++) {
            float left_mel = mel_lowFreq + bin * mel_freq_delta,
                center_mel = mel_lowFreq + (bin + 1) * mel_freq_delta,
                right_mel = mel_lowFreq + (bin + 2) * mel_freq_delta;

            if (vtln_warp_factor != 1.0) {
                left_mel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq,
                    vtln_warp_factor, left_mel);
                center_mel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq,
                    vtln_warp_factor, center_mel);
                right_mel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq,
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
            if (opts.htkMode && bin == 0 && mel_lowFreq != 0.0)
                bins_[bin].second.Set1D(0.0, 0);
        }
    }

    MelBanks::MelBanks(const MelBanks& other) :
        center_freqs_(other.center_freqs_),
        bins_(other.bins_),
        debug_(other.debug_),
        htkMode_(other.htkMode_) { }

    void MelBanks::Compute(const XTensor& power_spectrum,
        XTensor* mel_energies_out) const {
        INT32 numBins = bins_.size();
        ASSERT(mel_energies_out->GetDim(0) == numBins);

        for (INT32 i = 0; i < numBins; i++) {
            INT32 offset = bins_[i].first;
            const XTensor& v(bins_[i].second);
            XTensor partPowerSpectrum(&power_spectrum);
            int dimSize = { v.GetDim(0) };
            partPowerSpectrum.Resize(1, &dimSize);
            partPowerSpectrum.SetData(power_spectrum.GetCell(&offset, 1), v.GetDim(0), 0);
            XTensor tempReduce(ReduceSum(Multiply(v, partPowerSpectrum), 0));
            float energy = tempReduce.Get0D();
            // HTK-like flooring- for testing purposes (we prefer dither)
            if (htkMode_ && energy < 1.0) energy = 1.0;
            mel_energies_out->Set1D(energy, i);

            // The following assert was added due to a problem with OpenBlas that
            // we had at one point (it was a bug in that library).  Just to detect
            // it early.
            ASSERT(!std::isnan(mel_energies_out->Get1D(i)));
        }

    }
}