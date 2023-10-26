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
#include "utils.h"
#include "../niuTensor/tensor/core/CHeader.h"
using namespace nts;


namespace s2t {
    struct FrameExtractionOptions {
        float samp_freq;
        float frame_shift_ms;  // in milliseconds.
        float frame_length_ms;  // in milliseconds.
        float dither;  // Amount of dithering, 0.0 means no dither.
        float preemph_coeff;  // Preemphasis coefficient.
        bool remove_dc_offset;  // Subtract mean of wave before FFT.
        std::string window_type;  // e.g. Hamming window
        // May be "hamming", "rectangular", "povey", "hanning", "sine", "blackman"
        // "povey" is a window I made to be similar to Hamming but to go to zero at the
        // edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
        // I just don't think the Hamming window makes sense as a windowing function.
        bool round_to_power_of_two;
        float blackman_coeff;
        bool snip_edges;
        bool allow_downsample;
        bool allow_upsample;
        int max_feature_vectors;
        int torchPaddingLength; // In default, it will be half of frame length.
        std::string padMod;
        FrameExtractionOptions(const FrameExtractionOptions& opts ) {
            samp_freq = opts.samp_freq;
            frame_shift_ms = opts.frame_shift_ms;
            frame_length_ms = opts.frame_length_ms;
            dither = opts.dither;
            preemph_coeff = opts.preemph_coeff;
            remove_dc_offset = opts.remove_dc_offset;
            window_type = opts.window_type;
            snip_edges = opts.snip_edges;
            torchPaddingLength = opts.torchPaddingLength;
            round_to_power_of_two = opts.round_to_power_of_two;
            padMod = opts.padMod;
        }

        FrameExtractionOptions() :
            samp_freq(16000),
            frame_shift_ms(10.0),
            frame_length_ms(25.0),
            dither(0.0),
            preemph_coeff(0),
            remove_dc_offset(false),
            window_type("hanning_periodic"),
            round_to_power_of_two(false),
            blackman_coeff(0.42),
            snip_edges(true),
            allow_downsample(false),
            allow_upsample(false),
            max_feature_vectors(-1),
            torchPaddingLength(200),
            padMod("reflect")
        { }
        INT32 WindowShift() const {
            return static_cast<INT32>(samp_freq * 0.001 * frame_shift_ms);
        }
        INT32 WindowSize() const {
            return static_cast<INT32>(samp_freq * 0.001 * frame_length_ms);
        }
        INT32 PaddedWindowSize() const {
            if (round_to_power_of_two) {
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
            //return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) : WindowSize());
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

    void Preemphasize(XTensor waveform, float preemph_coeff);

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
            feature_window_function_(computer_.GetFrameOptions()) { }

        // Internal (and back-compatibility) interface for computing features, which
        // requires that the user has already checked that the sampling frequency
        // of the waveform is equal to the sampling frequency specified in
        // the frame-extraction options.
        void Compute(const XTensor& wave,
            float vtln_warp,
            XTensor* output);

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

        INT32 Dim() const { return computer_.Dim(); }

        // Copy constructor.
        OfflineFeatureTpl(const OfflineFeatureTpl<F>& other) :
            computer_(other.computer_),
            feature_window_function_(other.feature_window_function_) { }
    private:
        // Disallow assignment.
        OfflineFeatureTpl<F>& operator =(const OfflineFeatureTpl<F>& other);

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
        float new_sample_freq = computer_.GetFrameOptions().samp_freq;
        if (computer_.GetFrameOptions().torchPaddingLength != 0) {
            XTensor temp;
            int paddingDimSize = { wave.GetDim(0) + computer_.GetFrameOptions().torchPaddingLength * 2 };
            temp.Resize(1, &paddingDimSize);
            int index = { 0 };
            if (computer_.GetFrameOptions().padMod == "constant") {
                temp.SetZeroAll();
                temp.SetData(wave.GetCell(&index, 1), wave.GetDim(0), computer_.GetFrameOptions().torchPaddingLength);
            }
            else if (computer_.GetFrameOptions().padMod == "reflect") {
                temp.SetData(wave.GetCell(&index, 1), wave.GetDim(0), computer_.GetFrameOptions().torchPaddingLength);
                for (int i = 0; i <= computer_.GetFrameOptions().torchPaddingLength; i++) {
                    temp.Set1D(wave.Get1D(i), computer_.GetFrameOptions().torchPaddingLength - i);
                    temp.Set1D(wave.Get1D(wave.GetDim(0) - i - 1), temp.GetDim(0) - computer_.GetFrameOptions().torchPaddingLength + i - 1);
                    /*
                    std::cout << wave.Get1D(i) << " || " 
                        << temp.Get1D(computer_.GetFrameOptions().torchPaddingLength - i) << " || "
                        << computer_.GetFrameOptions().torchPaddingLength - i << " || "
                        << i << endl;
                    std::cout << wave.Get1D(wave.GetDim(0) - i - 1) << " || " 
                        << temp.Get1D(temp.GetDim(0) - computer_.GetFrameOptions().torchPaddingLength + i - 1) << " || "
                        << temp.GetDim(0) - computer_.GetFrameOptions().torchPaddingLength + i - 1 << " || "
                        << wave.GetDim(0) - i - 1 << endl;
                    */
                }
            }
            if(sample_freq == new_sample_freq) {
                Compute(temp, vtln_warp, output);
            }
            else {
                ASSERT(FALSE);
            }
        }
        else if (sample_freq == new_sample_freq) {
            Compute(wave, vtln_warp, output);
        }
        else {
            if (new_sample_freq < sample_freq &&
                !computer_.GetFrameOptions().samp_freq)
                ASSERT(FALSE);
                //ERR << "Waveform and config sample Frequency mismatch: "
                //<< sample_freq << " .vs " << new_sample_freq
                //<< " (use --allow-downsample=true to allow "
                //<< " downsampling the waveform).";
            else if (new_sample_freq > sample_freq &&
                !computer_.GetFrameOptions().samp_freq)
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
        XTensor temp = ReduceMax(ReduceMax(*output, 0), 0);
        float outputMax = temp.Get0D();
        ClipMe(*output, outputMax - 8, outputMax);
        ScaleAndShiftMe(*output, 1, 4);
        ScaleAndShiftMe(*output, 1 / 4, 0);
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
