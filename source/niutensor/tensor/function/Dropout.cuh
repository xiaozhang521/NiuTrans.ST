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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-12
 */

#ifndef __DROPOUT_CUH__
#define __DROPOUT_CUH__

#include "../XTensor.h"
#include "Loss.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* dropout function (Cuda version) */
void _CudaDropout(const XTensor * x, XTensor * y, const XTensor * r, DTYPE scaleFactor);

/* de/dx (Cuda version) */
void _CudaDropoutBackward(const XTensor * y, const XTensor * x,
                          const XTensor * dedy, XTensor * dedx,
                          const XTensor * mask, DTYPE scaleFactor);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __DROPOUT_CUH__