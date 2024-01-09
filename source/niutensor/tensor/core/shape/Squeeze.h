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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-27
 */

#ifndef __SQUEEZE_H__
#define __SQUEEZE_H__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* squeeze the tensor along the specified dimension */
void _Squeeze(XTensor * source, XTensor * target, int leadingDim = -1);

/* squeeze the tensor along the specified dimension (do it on site)
   keep the result in the input tensor a and return nothing */
void _SqueezeMe(XTensor * source, int leadingDim = -1);

/* squeeze the tensor along the specified dimension (do it on site)
   keep the result in the input tensor a and return nothing */
void SqueezeMe(XTensor & source, int leadingDim = -1);

/* squeeze the tensor along the specified dimension  (return an XTensor structure)
   make a new tensor to keep the result and return it */
XTensor Squeeze(XTensor & source, int leadingDim = -1);

void Squeeze(XTensor & source, XTensor & target, int leadingDim = -1);

} // namespace nts(NiuTrans.Tensor)

#endif // __SQUEEZE_H__