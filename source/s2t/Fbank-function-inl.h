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
#ifndef __FBANK_FUNCTION_INL_H__
#define __FBANK_FUNCTION_INL_H__

namespace s2t {

    template<typename Real> inline void ComplexAddProduct(const Real& a_re, const Real& a_im, const Real& b_re, const Real& b_im, Real* c_re, Real* c_im) {
        *c_re += b_re * a_re - b_im * a_im;
        *c_im += b_re * a_im + b_im * a_re;
    }

    template<typename Real> inline void ComplexImExp(Real x, Real* a_re, Real* a_im) {
        *a_re = std::cos(x);
        *a_im = std::sin(x);
    }
    
    template<typename Real> inline void ComplexMul(const Real& a_re, const Real& a_im,
        Real* b_re, Real* b_im) {
        Real tmp_re = (*b_re * a_re) - (*b_im * a_im);
        *b_im = *b_re * a_im + *b_im * a_re;
        *b_re = tmp_re;
    }


}


#endif // !__FBANK_FUNCTION_INL_H__
