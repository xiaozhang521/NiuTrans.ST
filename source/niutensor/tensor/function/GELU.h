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
* $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-09-21
*/

#ifndef __GELU_H__
#define __GELU_H__

#include "../XTensor.h"
#include <math.h>

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* rectify function y = max(0, x) */
void _GELU(const XTensor * x, XTensor * y);

/* rectify function y = max(0, x) (return an XTensor structure) */
XTensor GELU(const XTensor &x);

void GELU(const XTensor &x, XTensor &y);

/* de/dx */
void _GELUBackward(XTensor * y, XTensor * x, 
                   XTensor * dedy, XTensor * dedx);

} // namespace nts(NiuTrans.Tensor)

#endif // __GELU_H__