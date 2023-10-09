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

#ifndef __WAVE_LOADER__
#define __WAVE_LOADER__

#include <stdint.h>
#include "../niutensor/tensor/XTensor.h"
using namespace nts;

namespace s2t {
	const float kWaveSampleMax = 32768.0;

	class WaveInfo {
	public:
		WaveInfo() : samp_freq_(0), samp_count_(0),
			num_channels_(0), reverse_bytes_(0) {}

		/// Is stream size unknown? Duration and SampleCount not valid if true.
		bool IsStreamed() const { return samp_count_ < 0; }

		/// Sample frequency, Hz.
		float SampFreq() const { return samp_freq_; }

		/// Number of samples in stream. Invalid if IsStreamed() is true.
		UINT32 SampleCount() const { return samp_count_; }

		/// Approximate duration, seconds. Invalid if IsStreamed() is true.
		float Duration() const { return samp_count_ / samp_freq_; }

		/// Number of channels, 1 to 16.
		INT32 NumChannels() const { return num_channels_; }

		/// Bytes per sample.
		size_t BlockAlign() const { return 2 * num_channels_; }

		/// Wave data bytes. Invalid if IsStreamed() is true.
		size_t DataBytes() const { return samp_count_ * BlockAlign(); }

		/// Is data file byte order different from machine byte order?
		bool ReverseBytes() const { return reverse_bytes_; }

		/// 'is' should be opened in binary mode. Read() will throw on error.
		/// On success 'is' will be positioned at the beginning of wave data.
		void Read(std::istream& is);

	private:
		float samp_freq_;
		int samp_count_;     // 0 if empty, -1 if undefined length.
		UINT num_channels_;
		bool reverse_bytes_;   // File endianness differs from host.
	};

	// This class's purpose is to read in Wave files.
	class WaveData {
	public:
		WaveData(float samp_freq, const XTensor& data)
			: data_(data), samp_freq_(samp_freq) {}

		WaveData() : samp_freq_(0.0) {}

		/// Read() will throw on error.  It's valid to call Read() more than once--
		/// in this case it will destroy what was there before.
		/// "is" should be opened in binary mode.
		void Read(std::istream& is);

		/// Write() will throw on error.   os should be opened in binary mode.
		void Write(std::ostream& os) const;

		// This function returns the wave data-- it's in a matrix
		// because there may be multiple channels.  In the normal case
		// there's just one channel so Data() will have one row.
		const XTensor & Data() const { return data_; }

		float SampFreq() const { return samp_freq_; }

		// Returns the duration in seconds
		float Duration() const { return data_.GetDim(1) / samp_freq_; }

		void CopyFrom(const WaveData& other) {
			samp_freq_ = other.samp_freq_;
			data_.Resize(&other.data_);
		}

		void Clear() {
			data_.Resize(0, 0);
			samp_freq_ = 0.0;
		}

		void Swap(WaveData* other) {
			XTensor temp = data_;
			data_ = other->data_;
			other->data_ = temp;
			//data_.Swap(&(other->data_));
			std::swap(samp_freq_, other->samp_freq_);
		}

	private:
		static const UINT32 kBlockSize = 1024 * 1024;  // Use 1M bytes.
		XTensor data_;
		float samp_freq_;
	};

}


#endif //!__WAVE_LOADER__


#pragma once
