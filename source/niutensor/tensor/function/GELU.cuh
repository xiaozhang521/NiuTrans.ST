/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University. 
 * All rights reserved.
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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#ifndef __GELU_CUH__
#define __GELU_CUH__

#include "../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

#define M_PI       3.14159265358979323846   // pi
#define  M_1_SQRTPI      0.5641895835477563
#define  M_SQRT2         1.414213562373095145475
/* rectify function y = max(0, x) (Cuda version) */
void _CudaGELU(const XTensor * input, XTensor * output);

/* de/dx (Cuda version) */
void _CudaGELUBackward(XTensor * y, XTensor * x, 
                          XTensor * dedy, XTensor * dedx);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __GELU_CUH__