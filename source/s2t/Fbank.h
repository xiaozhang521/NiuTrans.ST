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
#include "S2TConfig.h"
#include <complex>
using namespace nts;

namespace s2t {

    

    

    // Including two options and a computer

    // MelBanksOptions is about how many MelBank in MelBankGroup and some other properties.
    struct MelBanksOptions {
        INT32 numBins;  // e.g. 25; number of triangular bins
        float lowFreq;  // e.g. 20; lower frequency cutoff
        float highFreq;  // an upper frequency cutoff; 0 -> no cutoff, negative
        // ->added to the Nyquist frequency to get the cutoff.
        float vtlnLow;  // vtln lower cutoff of warping function.
        float vtlnHigh;  // vtln upper cutoff of warping function: if negative, added
        // to the Nyquist frequency to get the cutoff.
        bool debugMel;
        // htkMode is a "hidden" config, it does not show up on command line.
        // Enables more exact compatibility with HTK, for testing purposes.  Affects
        // mel-energy flooring and reproduces a bug in HTK.
        bool htkMode;
        char customFilter[MAX_NAME_LEN];
        explicit MelBanksOptions(int numBins = 23)
            : numBins(numBins), lowFreq(20), highFreq(0), vtlnLow(100),
            vtlnHigh(-500), debugMel(false), htkMode(false), customFilter("../mel.csv") {}
    };

    struct FbankOptions {
        FrameExtractionOptions frameOpts;
        MelBanksOptions melOpts;
        bool useEnergy;  // append an extra dimension with energy to the filter banks
        float energyFloor;
        bool rawEnergy;  // If true, compute energy before preemphasis and windowing
        bool htkCompat;  // If true, put energy last (if using energy)
        bool useLogFbank;  // if true (default), produce log-filterbank, else linear
        bool usePower;  // if true (default), use power in filterbank analysis, else magnitude.
        bool oneSide;
        

        FbankOptions() : melOpts(80),
            useEnergy(false),
            energyFloor(0.0),
            rawEnergy(true),
            htkCompat(false),
            useLogFbank(true),
            oneSide(false),
            usePower(true) {}

        FbankOptions(S2TConfig& config)
        {
            useEnergy = config.extractor.useEnergy;
            energyFloor = config.extractor.energyFloor;
            
            rawEnergy = config.extractor.rawEnergy;
            htkCompat = config.extractor.htkCompat;
            useLogFbank = config.extractor.useLogFbank;
            usePower = config.extractor.usePower;
            oneSide = config.extractor.oneSide;

            std::strcpy(frameOpts.inputAudio, config.extractor.inputAudio);

            frameOpts.sampFreq = config.extractor.sampFreq;
            frameOpts.frameShiftMs = config.extractor.frameShiftMs;
            frameOpts.frameLengthMs = config.extractor.frameLengthMs;
            frameOpts.chunkLengthMs = config.extractor.chunkLengthMs;
            frameOpts.dither = config.extractor.dither;
            frameOpts.preemphCoeff = config.extractor.preemphCoeff;
            frameOpts.removeDcOffset = config.extractor.removeDcOffset;
            std::strcpy(frameOpts.windowType, config.extractor.windowType);

            frameOpts.roundToPowerOfTwo = config.extractor.roundToPowerOfTwo;
            frameOpts.blackmanCoeff = config.extractor.blackmanCoeff;
            frameOpts.snipEdges = config.extractor.snipEdges;
            frameOpts.allowDownsample = config.extractor.allowDownsample;
            frameOpts.allowUpsample = config.extractor.allowUpsample;
            frameOpts.maxFeatureVectors = config.extractor.maxFeatureVectors;
            frameOpts.torchPaddingLength = config.extractor.torchPaddingLength;
            std::strcpy(frameOpts.padMod, config.extractor.padMod);

            melOpts.numBins = config.extractor.numBins;
            melOpts.lowFreq = config.extractor.lowFreq;
            melOpts.highFreq = config.extractor.highFreq;
            melOpts.vtlnLow = config.extractor.vtlnLow;
            melOpts.vtlnHigh = config.extractor.vtlnHigh;
            melOpts.debugMel = config.extractor.debugMel;
            melOpts.htkMode = config.extractor.htkMode;
            std::strcpy(melOpts.customFilter, config.extractor.customFilter);
        }
        
        explicit FbankOptions(const FbankOptions& opts) : melOpts(opts.melOpts),
            frameOpts(opts.frameOpts),
            useEnergy(opts.useEnergy),
            energyFloor(opts.energyFloor),
            rawEnergy(opts.rawEnergy),
            htkCompat(opts.htkCompat),
            useLogFbank(opts.useLogFbank),
            oneSide(opts.oneSide),
            usePower(opts.usePower) {}
        
    };

    class MelBanks {
    public:

        static inline float InverseMelScale(float mel_freq) {
            return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
        }

        static inline float MelScale(float freq) {
            return 1127.0f * logf(1.0f + freq / 700.0f);
        }

        static float VtlnWarpFreq(float vtlnLow_cutoff,
            float vtlnHigh_cutoff,  // discontinuities in warp func
            float lowFreq,
            float highFreq,  // upper+lower frequency cutoffs in
            // the mel computation
            float vtln_warp_factor,
            float freq);

        static float VtlnWarpMelFreq(float vtlnLow_cutoff,
            float vtlnHigh_cutoff,
            float lowFreq,
            float highFreq,
            float vtln_warp_factor,
            float mel_freq);


        MelBanks(const MelBanksOptions& opts,
            const FrameExtractionOptions& frameOpts,
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

        // center frequencies of bins, numbered from 0 ... numBins-1.
        // Needed by GetCenterFreqs().
        XTensor center_freqs_;

        // the "bins_" vector is a vector, one for each bin, of a pair:
        // (the first nonzero fft-bin), (the XTensor of weights).
        std::vector<std::pair<INT32, XTensor > > bins_;
        

        bool debug_;
        bool htkMode_;
    };

    class FbankComputer {
    public:
        typedef FbankOptions Options;

        explicit FbankComputer(const FbankOptions& opts);
        FbankComputer(const FbankComputer& other);

        INT32 Dim() const {
            return opts_.melOpts.numBins + (opts_.useEnergy ? 1 : 0);
        }

        bool NeedRawLogEnergy() const { return opts_.useEnergy && opts_.rawEnergy; }

        const FrameExtractionOptions& GetFrameOptions() const {
            return opts_.frameOpts;
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
        float logEnergyFloor_;
        std::map<float, MelBanks*> mel_banks_;  // float is VTLN coefficient.
        SplitRadixRealFft<float>* srfft_;
        // Disallow assignment.
        FbankComputer& operator =(const FbankComputer& other);
    };

}
#endif