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

#ifndef INT_TO_int
#define INT_TO_int
#define INT16 int16_t
#define INT32 int32_t
#define INT64 int64_t
#define UINT16 int16_t
#define UINT32 uint32_t
#define UINT64 uint64_t
#define UINT uint32_t
#define TRUE true
#define FALSE false
#endif


namespace s2t {
	const float kWaveSampleMax = 32768.0;

	class WaveInfo {
	public:
		WaveInfo() : sampFreq_(0), samp_count_(0),
			num_channels_(0), reverse_bytes_(0) {}

		/// Is stream size unknown? Duration and SampleCount not valid if true.
		bool IsStreamed() const { return samp_count_ < 0; }

		/// Sample frequency, Hz.
		float SampFreq() const { return sampFreq_; }

		/// Number of samples in stream. Invalid if IsStreamed() is true.
		UINT32 SampleCount() const { return samp_count_; }

		/// Approximate duration, seconds. Invalid if IsStreamed() is true.
		float Duration() const { return samp_count_ / sampFreq_; }

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
		float sampFreq_;
		int samp_count_;     // 0 if empty, -1 if undefined length.
		UINT num_channels_;
		bool reverse_bytes_;   // File endianness differs from host.
	};

	// This class's purpose is to read in Wave files.
	class WaveData {
	public:
		WaveData(float sampFreq, const XTensor& data)
			: data_(data), sampFreq_(sampFreq) {}

		WaveData() : sampFreq_(0.0) {}

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

		float SampFreq() const { return sampFreq_; }

		// Returns the duration in seconds
		float Duration() const { return data_.GetDim(1) / sampFreq_; }

		void CopyFrom(const WaveData& other) {
			sampFreq_ = other.sampFreq_;
			data_.Resize(&other.data_);
		}

		void Clear() {
			data_.Resize(0, 0);
			sampFreq_ = 0.0;
		}

		void Swap(WaveData* other) {
			XTensor temp = data_;
			data_ = other->data_;
			other->data_ = temp;
			//data_.Swap(&(other->data_));
			std::swap(sampFreq_, other->sampFreq_);
		}

	private:
		static const UINT32 kBlockSize = 1024 * 1024;  // Use 1M bytes.
		XTensor data_;
		float sampFreq_;
	};

}


#endif //!__WAVE_LOADER__


#pragma once
