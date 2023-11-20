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
  * I want to use kaldi structure with our XTensor, but it is too difficult!
  * $Created by: HE Erfeng (heerfeng1023@gmail.com) 2023-09
  */

#ifndef __FEATURE_WINDOW__
#define __FEATURE_WINDOW__

#include <map>
#include <string>
#include <stdint.h>
#include "WaveLoader.h"
#include "utils.h"
#include "S2TConfig.h"
#include "../niuTensor/tensor/core/CHeader.h"
using namespace nts;


namespace s2t {
    struct FrameExtractionOptions {
        float sampFreq;
        float frameShiftMs;  // in milliseconds.
        float frameLengthMs;  // in milliseconds.
        float chunkLengthMs; // max audio length.
        float dither;  // Amount of dithering, 0.0 means no dither.
        float preemphCoeff;  // Preemphasis coefficient.
        bool removeDcOffset;  // Subtract mean of wave before FFT.
        char windowType[MAX_NAME_LEN];  // May be "hamming", "rectangular", "povey", "hanning", "sine", "blackman"
        bool roundToPowerOfTwo;
        float blackmanCoeff;
        bool snipEdges;
        bool allowDownsample;
        bool allowUpsample;
        int maxFeatureVectors;
        int torchPaddingLength; // In default, it will be half of frame length.
        char padMod[MAX_NAME_LEN];
        char inputAudio[MAX_NAME_LEN];
        FrameExtractionOptions(const FrameExtractionOptions& opts ) {
            sampFreq = opts.sampFreq;
            frameShiftMs = opts.frameShiftMs;
            frameLengthMs = opts.frameLengthMs;
            dither = opts.dither;
            preemphCoeff = opts.preemphCoeff;
            removeDcOffset = opts.removeDcOffset;
            *windowType = *opts.windowType;
            snipEdges = opts.snipEdges;
            torchPaddingLength = opts.torchPaddingLength;
            roundToPowerOfTwo = opts.roundToPowerOfTwo;
            *padMod = *opts.padMod;
            chunkLengthMs = opts.chunkLengthMs;
            *inputAudio = *opts.inputAudio;

        }

        FrameExtractionOptions() :
            sampFreq(16000),
            frameShiftMs(10.0),
            frameLengthMs(25.0),
            dither(0.0),
            preemphCoeff(0),
            removeDcOffset(false),
            windowType("hanning_periodic"),
            roundToPowerOfTwo(false),
            blackmanCoeff(0.42),
            snipEdges(true),
            allowDownsample(false),
            allowUpsample(false),
            maxFeatureVectors(-1),
            torchPaddingLength(200),
            padMod("reflect"),
            chunkLengthMs(30000.0)
        { }
        INT32 WindowShift() const {
            return static_cast<INT32>(sampFreq * 0.001 * frameShiftMs);
        }
        INT32 WindowSize() const {
            return static_cast<INT32>(sampFreq * 0.001 * frameLengthMs);
        }
        INT32 PaddedAudioSize() const {
            return static_cast<INT32>(sampFreq * 0.001 * chunkLengthMs);
        }
        INT32 PaddedWindowSize() const {
            if (roundToPowerOfTwo) {
                int n = WindowSize();
                ASSERT(n > 0);
                n--;
                n |= n >> 1;
                n |= n >> 2;
                n |= n >> 4;
                n |= n >> 8;
                n |= n >> 16;
                return n + 1;
            }
            else {
                return WindowSize();
            }
            //return (roundToPowerOfTwo ? RoundUpToNearestPowerOfTwo(WindowSize()) : WindowSize());
        }
    };
    struct FeatureWindowFunction {
        FeatureWindowFunction() {}
        explicit FeatureWindowFunction(const FrameExtractionOptions& opts);
        FeatureWindowFunction(const FeatureWindowFunction& other) :
            window(other.window) { }
        XTensor window;
    };
    /**
  This function returns the number of frames that we can extract from a wave
  file with the given number of samples in it (assumed to have the same
  sampling rate as specified in 'opts').

     @param [in] num_samples  The number of samples in the wave file.
     @param [in] opts     The frame-extraction options class

     @param [in] flush   True if we are asserting that this number of samples is
            'all there is', false if we expecting more data to possibly come
            in.  This only makes a difference to the answer if opts.snips_edges
            == false.  For offline feature extraction you always want flush ==
            true.  In an online-decoding context, once you know (or decide) that
            no more data is coming in, you'd call it with flush == true at the
            end to flush out any remaining data.
*/
    INT32 NumFrames(INT64 num_samples,
        const FrameExtractionOptions& opts,
        bool flush = true);

    /*
       This function returns the index of the first sample of the frame indexed
       'frame'.  If snip-edges=true, it just returns frame * opts.WindowShift(); if
       snip-edges=false, the formula is a little more complicated and the result may
       be negative.
    */
    INT64 FirstSampleOfFrame(INT32 frame,
        const FrameExtractionOptions& opts);



    void Dither(XTensor waveform, float dither_value);

    void Preemphasize(XTensor waveform, float preemphCoeff);

    /**
      This function does all the windowing steps after actually
      extracting the windowed signal: depending on the
      configuration, it does dithering, dc offset removal,
      preemphasis, and multiplication by the windowing function.
       @param [in] opts  The options class to be used
       @param [in] window_function  The windowing function-- should have
                        been initialized using 'opts'.
       @param [in,out] window  A vector of size opts.WindowSize().  Note:
          it will typically be a sub-vector of a larger vector of size
          opts.PaddedWindowSize(), with the remaining samples zero,
          as the FFT code is more efficient if it operates on data with
          power-of-two size.
       @param [out]   log_energy_pre_window If non-NULL, then after dithering and
          DC offset removal, this function will write to this pointer the log of
          the total energy (i.e. sum-squared) of the frame.
     */
    void ProcessWindow(const FrameExtractionOptions& opts,
        const FeatureWindowFunction& window_function,
        XTensor &window,
        float* log_energy_pre_window = NULL);


    /*
      ExtractWindow() extracts a windowed frame of waveform (possibly with a
      power-of-two, padded size, depending on the config), including all the
      proessing done by ProcessWindow().

      @param [in] sample_offset  If 'wave' is not the entire waveform, but
                       part of it to the left has been discarded, then the
                       number of samples prior to 'wave' that we have
                       already discarded.  Set this to zero if you are
                       processing the entire waveform in one piece, or
                       if you get 'no matching function' compilation
                       errors when updating the code.
      @param [in] wave  The waveform
      @param [in] f     The frame index to be extracted, with
                        0 <= f < NumFrames(sample_offset + wave.Dim(), opts, true)
      @param [in] opts  The options class to be used
      @param [in] window_function  The windowing function, as derived from the
                        options class.
      @param [out] window  The windowed, possibly-padded waveform to be
                         extracted.  Will be resized as needed.
      @param [out] log_energy_pre_window  If non-NULL, the log-energy of
                       the signal prior to pre-emphasis and multiplying by
                       the windowing function will be written to here.
    */
    void ExtractWindow(INT64 sample_offset,
        const XTensor wave,
        INT32 f,
        const FrameExtractionOptions& opts,
        const FeatureWindowFunction& window_function,
        XTensor &window,
        float* log_energy_pre_window = NULL);



    struct RandomState {
        RandomState() {
            seed = unsigned(rand()) + 27437;
        }
        unsigned seed;
    };

    //!!!!!!!!!!!!!!!multi process
    inline float RandUniform(struct RandomState* state = NULL) {
        return static_cast<float>((rand() + 1.0) / (RAND_MAX + 2.0));
    }

    inline float RandGauss(struct RandomState* state = NULL) {
        return static_cast<float>(sqrtf(-2 * logf(RandUniform(state)))
            * cosf(2 * M_PI * RandUniform(state)));
    }

    template <class F>
    class OfflineFeatureTpl {
    public:
        //typedef typename F::Options Options;
        typedef typename F Options;

        // Note: feature_window_function_ is the windowing function, which initialized
        // using the options class, that we cache at this level.
        OfflineFeatureTpl(const Options& opts) :
            computer_(opts),
            feature_window_function_(computer_.GetFrameOptions()) { 
            //computer_ = opts;
        }
        OfflineFeatureTpl() :{

        }
        // Internal (and back-compatibility) interface for computing features, which
        // requires that the user has already checked that the sampling frequency
        // of the waveform is equal to the sampling frequency specified in
        // the frame-extraction options.
        void Compute(const XTensor& wave,
            float vtln_warp,
            XTensor* output);

        void Compute(const XTensor& wave,
            float vtln_warp,
            XTensor* output,
            int noNeedComputeSize);

        // This const version of Compute() is a wrapper that
        // calls the non-const version on a temporary object.
        // It's less efficient than the non-const version.
        void Compute(const XTensor& wave,
            float vtln_warp,
            XTensor* output) const;

        /**
           Computes the features for one file (one sequence of features).
           This is the newer interface where you specify the sample frequency
           of the input waveform.
             @param [in] wave   The input waveform
             @param [in] sample_freq  The sampling frequency with which
                                      'wave' was sampled.
                                      if sample_freq is higher than the frequency
                                      specified in the config, we will downsample
                                      the waveform, but if lower, it's an error.
           @param [in] vtln_warp  The VTLN warping factor (will normally
                                  be 1.0)
           @param [out]  output  The matrix of features, where the row-index
                                 is the frame index.
        */
        void ComputeFeatures(const XTensor& wave,
            float sample_freq,
            float vtln_warp,
            XTensor* output);
        void Read() {
            ifstream inFile(computer_.GetFrameOptions().inputAudio, ios::in | ios::binary);
            if (!inFile) {
                cout << "Error: no inputAudio file" << endl;
            }
            data_.Read(inFile);
        }

        const class WaveData & Data() const {
            return data_;
        }

        INT32 Dim() const { return computer_.Dim(); }

        // Copy constructor.
        OfflineFeatureTpl(const OfflineFeatureTpl<F>& other) :
            computer_(other.computer_),
            feature_window_function_(other.feature_window_function_) { }
    private:
        // Disallow assignment.
        // OfflineFeatureTpl<F>& operator =(const OfflineFeatureTpl<F>& other);
        WaveData data_;
        F computer_;
        FeatureWindowFunction feature_window_function_;
    };

    template <class F>
    void OfflineFeatureTpl<F>::ComputeFeatures(
        const XTensor& wave,
        float sample_freq,
        float vtln_warp,
        XTensor* output) {
        ASSERT(output != NULL);
        float new_sample_freq = computer_.GetFrameOptions().sampFreq;
        INT32 paddedAudioSize = { computer_.GetFrameOptions().PaddedAudioSize() };
        XTensor paddedWave;
        XTensor torchPaddedWave;
        int startIndex = { 0 };
        int noNeedComputeSize = 0;
        if (paddedAudioSize > 0) {
            if ((paddedAudioSize - wave.GetDim(0)) > computer_.GetFrameOptions().WindowSize()) {
                paddedAudioSize = wave.GetDim(0) + computer_.GetFrameOptions().WindowSize();
                noNeedComputeSize = computer_.GetFrameOptions().PaddedAudioSize() - paddedAudioSize;
                paddedWave.Resize(1, &paddedAudioSize);
                paddedWave.SetData(wave.GetCell(&startIndex, 1), wave.GetDim(0), 0);
            }
            else paddedWave = SelectRange(wave, 0, 0, paddedAudioSize);
        }
        else { 
            paddedAudioSize = wave.GetDim(0); 
            paddedWave.Resize(1, &paddedAudioSize);
            paddedWave.SetData(wave.GetCell(&startIndex, 1), wave.GetDim(0), 0);
        }
        

        if (computer_.GetFrameOptions().torchPaddingLength != 0) {
            
            int torchDimSize = { paddedAudioSize + computer_.GetFrameOptions().torchPaddingLength * 2 };
            torchPaddedWave.Resize(1, &torchDimSize);
            torchPaddedWave.SetZeroAll();
            torchPaddedWave.SetData(paddedWave.GetCell(&startIndex, 1), paddedWave.GetDim(0), computer_.GetFrameOptions().torchPaddingLength);
            
            
            if (strcmp(computer_.GetFrameOptions().padMod,"reflect") == 0) {
                for (int i = 0; i <= computer_.GetFrameOptions().torchPaddingLength; i++) {
                    torchPaddedWave.Set1D(paddedWave.Get1D(i), computer_.GetFrameOptions().torchPaddingLength - i);
                    torchPaddedWave.Set1D(paddedWave.Get1D(paddedWave.GetDim(0) - i - 1), torchPaddedWave.GetDim(0) - computer_.GetFrameOptions().torchPaddingLength + i - 1);
                    /*
                    std::cout << paddedWave.Get1D(i) << " || " 
                        << torchPaddedWave.Get1D(computer_.GetFrameOptions().torchPaddingLength - i) << " || "
                        << computer_.GetFrameOptions().torchPaddingLength - i << " || "
                        << i << endl;
                    std::cout << paddedWave.Get1D(paddedWave.GetDim(0) - i - 1) << " || " 
                        << torchPaddedWave.Get1D(torchPaddedWave.GetDim(0) - computer_.GetFrameOptions().torchPaddingLength + i - 1) << " || "
                        << torchPaddedWave.GetDim(0) - computer_.GetFrameOptions().torchPaddingLength + i - 1 << " || "
                        << paddedWave.GetDim(0) - i - 1 << endl;
                    */
                }
            }
            
            if(sample_freq == new_sample_freq) {
                Compute(torchPaddedWave, vtln_warp, output, noNeedComputeSize);
            }
            else {
                ASSERT(sample_freq == new_sample_freq);
            }
        }
        else if (sample_freq == new_sample_freq) {
            Compute(paddedWave, vtln_warp, output);
        }
        else {
            if (new_sample_freq < sample_freq &&
                !computer_.GetFrameOptions().sampFreq)
                ASSERT(FALSE);
                //ERR << "Waveform and config sample Frequency mismatch: "
                //<< sample_freq << " .vs " << new_sample_freq
                //<< " (use --allow-downsample=true to allow "
                //<< " downsampling the waveform).";
            else if (new_sample_freq > sample_freq &&
                !computer_.GetFrameOptions().sampFreq)
                ASSERT(FALSE);
                //ERR << "Waveform and config sample Frequency mismatch: "
                //<< sample_freq << " .vs " << new_sample_freq
                //<< " (use --allow-upsample=true option to allow "
                //<< " upsampling the waveform).";
            // Resample the waveform.
            XTensor resampled_wave(wave);
            //!!!!!!!!!!!!!!!!!!!!!!!!
            //ResampleWaveform(sample_freq, wave,
            //    new_sample_freq, &resampled_wave);
            Compute(resampled_wave, vtln_warp, output);
        }
    }

    template <class F>
    void OfflineFeatureTpl<F>::Compute(
        const XTensor& wave,
        float vtln_warp,
        XTensor* output,
        int noNeedComputeSize) {
        ASSERT(output != NULL);

        // This is the final dimension 
        
        INT32 rows_out = NumFrames(wave.GetDim(0), computer_.GetFrameOptions());
        INT32 cols_out = computer_.Dim();
        INT32 paddedRows = rows_out;
        if (noNeedComputeSize != 0) {
            paddedRows = NumFrames(wave.GetDim(0) + noNeedComputeSize, computer_.GetFrameOptions());
        }
        if (rows_out == 0) {
            output->Reshape(0, 0);
            return;
        }
        int dimSize[] = { paddedRows, cols_out };
        output->Resize(2, dimSize, X_FLOAT, 1.0);
        output->SetZeroAll();
        if (computer_.NeedLog10()) {
            ClipMe(*output, 1e-10, FLT_MAX);
            Log10Me(*output);
        }
        XTensor window;  // windowed waveform.
        bool use_raw_log_energy = computer_.NeedRawLogEnergy();
        int startIndex[] = { 0, 0 };
        for (INT32 r = 0; r < rows_out; r++) {  // r is frame index.
            float raw_log_energy = 0.0;
            ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                feature_window_function_, window, (use_raw_log_energy ? &raw_log_energy : NULL));

            //int rowIndex = { r, 0 };
            XTensor output_row;
            int rowDimSize = { rows_out };
            output_row.Resize(1, &rowDimSize);
            computer_.Compute(raw_log_energy, vtln_warp, &window, output_row);
            output->SetData(output_row.GetCell(startIndex, 2), cols_out, r * cols_out);
        }
        *output = SelectRange(*output, 0, 0, paddedRows - 1);
        // int newDimSize[] = { rows_out - 1 , cols_out };
        // output->SetDim(newDimSize);
        XTensor temp = ReduceMax(ReduceMax(*output, 0), 0);
        float outputMax = temp.Get0D();
        ClipMe(*output, outputMax - 8, outputMax);
        ScaleAndShiftMe(*output, 1.0, 4.0);
        ScaleAndShiftMe(*output, 0.25, 0.0);
    }

    template <class F>
    void OfflineFeatureTpl<F>::Compute(
        const XTensor& wave,
        float vtln_warp,
        XTensor* output) {
        ASSERT(output != NULL);

        // This is the final dimension 
        INT32 rows_out = NumFrames(wave.GetDim(0), computer_.GetFrameOptions());
        INT32 cols_out = computer_.Dim();

        if (rows_out == 0) {
            output->Reshape(0,0);
            return;
        }
        int dimSize[] = { rows_out, cols_out };
        output->Resize(2, dimSize, X_FLOAT, 1.0);
        XTensor window;  // windowed waveform.
        bool use_raw_log_energy = computer_.NeedRawLogEnergy();
        int startIndex[] = {0, 0};
        for (INT32 r = 0; r < rows_out; r++) {  // r is frame index.
            float raw_log_energy = 0.0;
            ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                feature_window_function_, window, (use_raw_log_energy ? &raw_log_energy : NULL));

            //int rowIndex = { r, 0 };
            XTensor output_row;
            int rowDimSize = { rows_out };
            output_row.Resize(1, &rowDimSize);
            computer_.Compute(raw_log_energy, vtln_warp, &window, output_row);
            output->SetData(output_row.GetCell(startIndex, 2), cols_out, r * cols_out);
        }
        *output = SelectRange(*output, 0, 0, rows_out - 1);
        // int newDimSize[] = { rows_out - 1 , cols_out };
        // output->SetDim(newDimSize);
        XTensor temp = ReduceMax(ReduceMax(*output, 0), 0);
        float outputMax = temp.Get0D();
        ClipMe(*output, outputMax - 8, outputMax);
        ScaleAndShiftMe(*output, 1.0, 4.0);
        ScaleAndShiftMe(*output, 0.25, 0.0);
    }

    template <class F>
    void OfflineFeatureTpl<F>::Compute(
        const XTensor& wave,
        float vtln_warp,
        XTensor* output) const {
        OfflineFeatureTpl<F> temp(*this);
        // call the non-const version of Compute() on a temporary copy of this object.
        // This is a workaround for const-ness that may sometimes be useful in
        // multi-threaded code, although it's not optimally efficient.
        temp.Compute(wave, vtln_warp, output);
    }

}





#endif //endif __FEATURE_WINDOW__
