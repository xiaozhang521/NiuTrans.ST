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

#include "FeatureWindow.h"
#include "utils.h"
#include <stdint.h>
#include "..\niutensor\tensor\core\reduce\ReduceSum.h"
#include "..\niutensor\tensor\core\math\ScaleAndShift.h"
#include "../niutensor/tensor/core/arithmetic/Multiply.h"
using namespace nts;

namespace s2t {
    FeatureWindowFunction::FeatureWindowFunction(const FrameExtractionOptions& opts) {
        //INT32 frame_length = opts.WindowSize();
        INT32 frame_length = opts.WindowSize();
        ASSERT(frame_length > 0);
        const int dimSize[] = { frame_length };
        window.Resize(1, dimSize);
        double a = M_2PI / (frame_length - 1);
        for (INT32 i = 0; i < frame_length; i++) {
            double i_fl = static_cast<double>(i);
            if (opts.window_type == "hanning") {
                if (!window.Set1D(0.5 - 0.5 * cos(a * i_fl), i)) {
                    ASSERT(FALSE);
                }
            }
            else if (opts.window_type == "sine") {
                // when you are checking ws wikipedia, please
                // note that 0.5 * a = M_PI/(frame_length-1)
                if (!window.Set1D(sin(0.5 * a * i_fl), i)) {
                    ASSERT(FALSE);
                }
            }
            else if (opts.window_type == "hamming") {
                if (!window.Set1D(0.54 - 0.46 * cos(a * i_fl), i)) {
                    ASSERT(FALSE);
                }
            }
            else if (opts.window_type == "povey") {  // like hamming but goes to zero at edges.
                if (!window.Set1D(pow(0.5 - 0.5 * cos(a * i_fl), 0.85), i)) {
                    ASSERT(FALSE);
                }
            }
            else if (opts.window_type == "rectangular") {
                if (!window.Set1D(1.0, i)) {
                    ASSERT(FALSE);
                }
            }
            else if (opts.window_type == "blackman") {
                if (!window.Set1D(opts.blackman_coeff - 0.5 * cos(a * i_fl) +
                    (0.5 - opts.blackman_coeff) * cos(2 * a * i_fl), i)) {
                    ASSERT(FALSE);
                }
            }
            else {
                ASSERT(FALSE);
            }
        }
    }

    INT64 FirstSampleOfFrame(INT32 frame,
        const FrameExtractionOptions& opts) {
        INT64 frame_shift = opts.WindowShift();
        if (opts.snip_edges) {
            return frame * frame_shift;
        }
        else {
            INT64 midpoint_of_frame = frame_shift * frame + frame_shift / 2,
                beginning_of_frame = midpoint_of_frame - opts.WindowSize() / 2;
            return beginning_of_frame;
        }
    }


    void Dither(XTensor* waveform, float dither_value) {
        if (dither_value == 0.0)
            return;
        INT32 dim = waveform->GetDim(0);
        int index[] = { 0 };
        float* data = reinterpret_cast<float*>(waveform->GetCell(index, 1));
        RandomState rstate;
        for (INT32 i = 0; i < dim; i++)
            data[i] += RandGauss(&rstate) * dither_value;
    }


    void Preemphasize(XTensor* waveform, float preemph_coeff) {
        if (preemph_coeff == 0.0) return;
        ASSERT(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
        for (INT32 i = waveform->GetDim(0) - 1; i > 0; i--)
            waveform->Set1D(waveform->Get1D(i) - preemph_coeff * waveform->Get1D(i - 1), i);
        waveform->Set1D(waveform->Get1D(0) - preemph_coeff * waveform->Get1D(0), 0);
    }

    INT32 NumFrames(INT64 num_samples,
        const FrameExtractionOptions& opts,
        bool flush) {
        INT64 frame_shift = opts.WindowShift();
        INT64 frame_length = opts.WindowSize();
        if (opts.snip_edges) {
            // with --snip-edges=true (the default), we use a HTK-like approach to
            // determining the number of frames-- all frames have to fit completely into
            // the waveform, and the first frame begins at sample zero.
            if (num_samples < frame_length)
                return 0;
            else
                return (1 + ((num_samples - frame_length) / frame_shift));
            // You can understand the expression above as follows: 'num_samples -
            // frame_length' is how much room we have to shift the frame within the
            // waveform; 'frame_shift' is how much we shift it each time; and the ratio
            // is how many times we can shift it (integer arithmetic rounds down).
        }
        else {
            // if --snip-edges=false, the number of frames is determined by rounding the
            // (file-length / frame-shift) to the nearest integer.  The point of this
            // formula is to make the number of frames an obvious and predictable
            // function of the frame shift and signal length, which makes many
            // segmentation-related questions simpler.
            //
            // Because integer division in C++ rounds toward zero, we add (half the
            // frame-shift minus epsilon) before dividing, to have the effect of
            // rounding towards the closest integer.
            INT32 num_frames = (num_samples + (frame_shift / 2)) / frame_shift;

            if (flush)
                return num_frames;

            // note: 'end' always means the last plus one, i.e. one past the last.
            INT64 end_sample_of_last_frame = FirstSampleOfFrame(num_frames - 1, opts)
                + frame_length;

            // the following code is optimized more for clarity than efficiency.
            // If flush == false, we can't output frames that extend past the end
            // of the signal.
            while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
                num_frames--;
                end_sample_of_last_frame -= frame_shift;
            }
            return num_frames;
        }
    }

    void ProcessWindow(const FrameExtractionOptions& opts,
        const FeatureWindowFunction& window_function,
        XTensor window,
        float* log_energy_pre_window) {
        INT32 frame_length = opts.WindowSize();
        ASSERT(window.GetDim(0) == frame_length);

        if (opts.dither != 0.0)
            Dither(&window, opts.dither);

        if (opts.remove_dc_offset) {
            XTensor temp(ReduceSum(window, 0));
            //temp = ReduceSum(window, 1);
            double dcOffset = temp.Get0D() / frame_length;
            //temp.Set0D(dcOffset);
            window = ScaleAndShift(window, 1.0F, -dcOffset);
            //window.operator-(temp);
        }

        if (log_energy_pre_window != NULL) {
            // The dot oprator of XTensor.
            XTensor sq(ReduceSum(Multiply(window, window, 0), 0));
            float energy = std::max<float>(sq.Get0D(),
                std::numeric_limits<float>::epsilon());
            *log_energy_pre_window = logf(energy);
        }

        if (opts.preemph_coeff != 0.0)
            Preemphasize(&window, opts.preemph_coeff);

        window = window.operator*(window_function.window);
    }


    // ExtractWindow extracts a windowed frame of waveform with a power-of-two,
    // padded size.  It does mean subtraction, pre-emphasis and dithering as
    // requested.
    void ExtractWindow(INT64 sample_offset,
        const XTensor wave,
        INT32 f,  // with 0 <= f < NumFrames(feats, opts)
        const FrameExtractionOptions& opts,
        const FeatureWindowFunction& window_function,
        XTensor window,
        float* log_energy_pre_window) {
        ASSERT(sample_offset >= 0 && wave.GetDim(0) != 0);
        INT32 frame_length = opts.WindowSize(),
            frame_length_padded = opts.PaddedWindowSize();
        INT64 num_samples = sample_offset + wave.GetDim(0),
            start_sample = FirstSampleOfFrame(f, opts),
            end_sample = start_sample + frame_length;

        if (opts.snip_edges) {
            ASSERT(start_sample >= sample_offset &&
                end_sample <= num_samples);
        }
        else {
            ASSERT(sample_offset == 0 || start_sample >= sample_offset);
        }
        if (window.order != 1) {
            int dimSize[] = { frame_length_padded };
            window.Resize(1, dimSize);
        }
        if (window.GetDim(0) != frame_length_padded) {
            int dimSize[] = { frame_length_padded };
            window.Resize(1,dimSize);
        }
            

        // wave_start and wave_end are start and end indexes into 'wave', for the
        // piece of wave that we're trying to extract.
        INT32 wave_start = INT32(start_sample - sample_offset),
            wave_end = wave_start + frame_length;
        if (wave_start >= 0 && wave_end <= wave.GetDim(0)) {
            // the normal case-- no edge effects to consider.
            int index[] = { wave_start };
            window.SetData(wave.GetCell(index, 1), frame_length, 0);
        }
        else {
            // Deal with any end effects by reflection, if needed.  This code will only
            // be reached for about two frames per utterance, so we don't concern
            // ourselves excessively with efficiency.
            INT32 wave_dim = wave.GetDim(0);
            for (INT32 s = 0; s < frame_length; s++) {
                INT32 s_in_wave = s + wave_start;
                while (s_in_wave < 0 || s_in_wave >= wave_dim) {
                    // reflect around the beginning or end of the wave.
                    // e.g. -1 -> 0, -2 -> 1.
                    // dim -> dim - 1, dim + 1 -> dim - 2.
                    // the code supports repeated reflections, although this
                    // would only be needed in pathological cases.
                    if (s_in_wave < 0) s_in_wave = -s_in_wave - 1;
                    else s_in_wave = 2 * wave_dim - 1 - s_in_wave;
                }
                window.Set1D(wave.Get1D(s_in_wave), s);
            }
        }
        int index[] = { 0 };
        if (frame_length_padded > frame_length) {
            XTensor zeroPart(window);
            int dimSize[] = { frame_length_padded - frame_length };
            zeroPart.SetDim(dimSize);
            zeroPart.SetZeroAll();
            window.SetData(zeroPart.GetCell(index, 1), frame_length_padded - frame_length, frame_length);
        }

        XTensor frame(window);
        int frameDimSize[] = { frame_length };
        frame.Resize(1, frameDimSize);
        frame.SetData(window.GetCell(index, 1), frame_length, 0);
        ProcessWindow(opts, window_function, frame, log_energy_pre_window);
    }

    signed int RoundUpToNearestPowerOfTwo(signed int n) {
        ASSERT(n > 0);
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }

}