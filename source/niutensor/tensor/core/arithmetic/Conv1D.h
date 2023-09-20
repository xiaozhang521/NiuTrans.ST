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
 * $Created by: Yuhao Zhang (email: hello_master1954@163.com) 2023-09-20
 */

#ifndef __CONV_H__
#define __CONV_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
1D Convolution
*/
void _Conv1DBase(const XTensor *input, const XTensor *weight, const XTensor *bias, XTensor *c,
                 int stride=1, int padding=0, bool useBias=true);

/* 
1D Convolution bias (return an XTensor structure)
make a new tensor to keep the result and return it
*/
XTensor Conv1DBias(const XTensor &input, const XTensor &weight, const XTensor &bias,
                   int stride=1, int padding=0, bool useBias=true);
/*
1D Convolution (return an XTensor structure)
make a new tensor to keep the result and return it
*/
XTensor Conv1DBase(const XTensor& input, const XTensor& weight, int stride, int padding);

#define CheckCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      fprintf(stderr,"cudnn error: %s[%d] %s\n",__FILE__, __LINE__,cudnnGetErrorString(status)); \
      exit(-1);                                              \
    }                                                        \
  }

} // namespace nts(NiuTrans.Tensor)

#endif // __CONV_H__