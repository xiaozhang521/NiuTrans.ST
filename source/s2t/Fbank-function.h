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
  * $Created by: HE Erfeng (heerfeng1023@gmail.com) 2023-10
  */

#ifndef __FBANK_FUNCTION__
#define __FBANK_FUNCTION__
#include <stdint.h>
#include "../niutensor/tensor/XTensor.h"
#include <vector>
#include "utils.h"
#include <complex>
#include <cstring>

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

using namespace nts;

namespace s2t {

    template<typename Real> void RealFft(XTensor* v, bool forward);

    template<typename Real> void ComplexFft(XTensor* v, bool forward, XTensor* tmp_work = NULL);

    template<typename Real> void OneSidedFFT(XTensor* v, bool forward);

    std::vector<std::complex<float>> OneSidedFFT(const std::vector<float>& signal);

    template<typename Real>
    class SplitRadixComplexFft {
    public:

        // N is the number of complex points (must be a power of two, or this
        // will crash).  Note that the constructor does some work so it's best to
        // initialize the object once and do the computation many times.
        SplitRadixComplexFft(INT32 N);

        // Copy constructor
        SplitRadixComplexFft(const SplitRadixComplexFft& other);

        // Does the FFT computation, given pointers to the real and
        // imaginary parts.  If "forward", do the forward FFT; else
        // do the inverse FFT (without the 1/N factor).
        // xr and xi are pointers to zero-based arrays of size N,
        // containing the real and imaginary parts
        // respectively.
        void Compute(Real* xr, Real* xi, bool forward) const;

        // This version of Compute takes a single array of size N*2,
        // containing [ r0 im0 r1 im1 ... ].  Otherwise its behavior is  the
        // same as the version above.
        void Compute(Real* x, bool forward);


        // This version of Compute is const; it operates on an array of size N*2
        // containing [ r0 im0 r1 im1 ... ], but it uses the argument "temp_buffer" as
        // temporary storage instead of a class-member variable.  It will allocate it if
        // needed.
        void Compute(Real* x, bool forward, std::vector<Real>* temp_buffer) const;

        ~SplitRadixComplexFft();

    protected:
        // temp_buffer_ is allocated only if someone calls Compute with only one Real*
        // argument and we need a temporary buffer while creating interleaved data.
        std::vector<Real> temp_buffer_;
    private:
        void ComputeTables();
        void ComputeRecursive(Real* xr, Real* xi, INT32 logn) const;
        void BitReversePermute(Real* x, INT32 logn) const;

        INT32 N_;
        INT32 logn_;  // log(N)

        INT32* brseed_;
        // brseed is Evans' seed table, ref:  (Ref: D. M. W.
        // Evans, "An improved digit-reversal permutation algorithm ...",
        // IEEE Trans. ASSP, Aug. 1987, pp. 1120-1125).
        Real** tab_;       // Tables of butterfly coefficients.

        // Disallow assignment.
        SplitRadixComplexFft& operator =(const SplitRadixComplexFft<Real>& other);
    };

    template<typename Real>
    class SplitRadixRealFft : private SplitRadixComplexFft<Real> {
    public:
        SplitRadixRealFft(INT32 N) :  // will fail unless N>=4 and N is a power of 2.
            SplitRadixComplexFft<Real>(N / 2), N_(N) { }

        // Copy constructor
        SplitRadixRealFft(const SplitRadixRealFft<Real>& other) :
            SplitRadixComplexFft<Real>(other), N_(other.N_) { }

        /// If forward == true, this function transforms from a sequence of N real points to its complex fourier
        /// transform; otherwise it goes in the reverse direction.  If you call it
        /// in the forward and then reverse direction and multiply by 1.0/N, you
        /// will get back the original data.
        /// The interpretation of the complex-FFT data is as follows: the array
        /// is a sequence of complex numbers C_n of length N/2 with (real, im) format,
        /// i.e. [real0, real_{N/2}, real1, im1, real2, im2, real3, im3, ...].
        void Compute(Real* x, bool forward);


        /// This is as the other Compute() function, but it is a const version that
        /// uses a user-supplied buffer.
        void Compute(Real* x, bool forward, std::vector<Real>* temp_buffer) const;

    private:
        // Disallow assignment.
        SplitRadixRealFft& operator =(const SplitRadixRealFft<Real>& other);
        int N_;
    };
    

}

#include "Fbank-function-inl.h"

#endif // !__FBANK_FUNCTION__
