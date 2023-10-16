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
  * filterbank!
  * $Created by: HE Erfeng (heerfeng1023@gmail.com) 2023-10
  */

#ifndef __FBANK__
#define __FBANK__

#include <stdint.h>
#include "../niutensor/tensor/XTensor.h"
#include "FeatureWindow.h"
#include <vector>
#include "Fbank-function.h"
#include "Fbank-function-inl.h"
using namespace nts;

namespace s2t {

    

    

    // Including two options and a computer

    // MelBanksOptions is about how many MelBank in MelBankGroup and some other properties.
    struct MelBanksOptions {
        INT32 num_bins;  // e.g. 25; number of triangular bins
        float low_freq;  // e.g. 20; lower frequency cutoff
        float high_freq;  // an upper frequency cutoff; 0 -> no cutoff, negative
        // ->added to the Nyquist frequency to get the cutoff.
        float vtln_low;  // vtln lower cutoff of warping function.
        float vtln_high;  // vtln upper cutoff of warping function: if negative, added
        // to the Nyquist frequency to get the cutoff.
        bool debug_mel;
        // htk_mode is a "hidden" config, it does not show up on command line.
        // Enables more exact compatibility with HTK, for testing purposes.  Affects
        // mel-energy flooring and reproduces a bug in HTK.
        bool htk_mode;
        explicit MelBanksOptions(int num_bins = 23)
            : num_bins(num_bins), low_freq(20), high_freq(0), vtln_low(100),
            vtln_high(-500), debug_mel(false), htk_mode(false) {}
    };

    struct FbankOptions {
        FrameExtractionOptions frame_opts;
        MelBanksOptions mel_opts;
        bool use_energy;  // append an extra dimension with energy to the filter banks
        float energy_floor;
        bool raw_energy;  // If true, compute energy before preemphasis and windowing
        bool htk_compat;  // If true, put energy last (if using energy)
        bool use_log_fbank;  // if true (default), produce log-filterbank, else linear
        bool use_power;  // if true (default), use power in filterbank analysis, else magnitude.

        FbankOptions() : mel_opts(23),
            // defaults the #mel-banks to 23 for the FBANK computations.
            // this seems to be common for 16khz-sampled data,
            // but for 8khz-sampled data, 15 may be better.
            use_energy(false),
            energy_floor(0.0),
            raw_energy(true),
            htk_compat(false),
            use_log_fbank(true),
            use_power(true) {}
    };

    class MelBanks {
    public:

        static inline float InverseMelScale(float mel_freq) {
            return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
        }

        static inline float MelScale(float freq) {
            return 1127.0f * logf(1.0f + freq / 700.0f);
        }

        static float VtlnWarpFreq(float vtln_low_cutoff,
            float vtln_high_cutoff,  // discontinuities in warp func
            float low_freq,
            float high_freq,  // upper+lower frequency cutoffs in
            // the mel computation
            float vtln_warp_factor,
            float freq);

        static float VtlnWarpMelFreq(float vtln_low_cutoff,
            float vtln_high_cutoff,
            float low_freq,
            float high_freq,
            float vtln_warp_factor,
            float mel_freq);


        MelBanks(const MelBanksOptions& opts,
            const FrameExtractionOptions& frame_opts,
            float vtln_warp_factor);

        /// Compute Mel energies (note: not log enerties).
        /// At input, "fft_energies" contains the FFT energies (not log).
        void Compute(const XTensor& fft_energies,
            XTensor* mel_energies_out) const;

        INT32 NumBins() const { return bins_.size(); }

        // returns vector of central freq of each bin; needed by plp code.
        const XTensor& GetCenterFreqs() const { return center_freqs_; }

        const std::vector<std::pair<INT32, XTensor > >& GetBins() const {
            return bins_;
        }

        // Copy constructor
        MelBanks(const MelBanks& other);
    private:
        // Disallow assignment
        MelBanks& operator = (const MelBanks& other);

        // center frequencies of bins, numbered from 0 ... num_bins-1.
        // Needed by GetCenterFreqs().
        XTensor center_freqs_;

        // the "bins_" vector is a vector, one for each bin, of a pair:
        // (the first nonzero fft-bin), (the XTensor of weights).
        std::vector<std::pair<INT32, XTensor > > bins_;
        

        bool debug_;
        bool htk_mode_;
    };

    class FbankComputer {
    public:
        typedef FbankOptions Options;

        explicit FbankComputer(const FbankOptions& opts);
        FbankComputer(const FbankComputer& other);

        INT32 Dim() const {
            return opts_.mel_opts.num_bins + (opts_.use_energy ? 1 : 0);
        }

        bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

        const FrameExtractionOptions& GetFrameOptions() const {
            return opts_.frame_opts;
        }

        /**
           Function that computes one frame of features from
           one frame of signal.

           @param [in] signal_raw_log_energy The log-energy of the frame of the signal
               prior to windowing and pre-emphasis, or
               log(numeric_limits<float>::min()), whichever is greater.  Must be
               ignored by this function if this class returns false from
               this->NeedsRawLogEnergy().
           @param [in] vtln_warp  The VTLN warping factor that the user wants
               to be applied when computing features for this utterance.  Will
               normally be 1.0, meaning no warping is to be done.  The value will
               be ignored for feature types that don't support VLTN, such as
               spectrogram features.
           @param [in] signal_frame  One frame of the signal,
             as extracted using the function ExtractWindow() using the options
             returned by this->GetFrameOptions().  The function will use the
             vector as a workspace, which is why it's a non-const pointer.
           @param [out] feature  Pointer to a vector of size this->Dim(), to which
               the computed feature will be written.
        */
        void Compute(float signal_raw_log_energy,
            float vtln_warp,
            XTensor* signal_frame,
            XTensor &feature);

        ~FbankComputer();

    private:
        const MelBanks* GetMelBanks(float vtln_warp);


        FbankOptions opts_;
        float log_energy_floor_;
        std::map<float, MelBanks*> mel_banks_;  // float is VTLN coefficient.
        SplitRadixRealFft<float>* srfft_;
        // Disallow assignment.
        FbankComputer& operator =(const FbankComputer& other);
    };

}
#endif