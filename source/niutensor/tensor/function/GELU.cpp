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

#include "../XName.h"
#include "../core/shape/IsSameShaped.h"
#include "GELU.h"
#include "GELU.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
GLUE function y = 0.5x+(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
>> x - input tensor
>> y - output tensor
*/
void _GELU(const XTensor * x, XTensor * y)
{
    CheckNTErrors(_IsSameShaped(x, y), 
                 "The input tensor and output tensor must have the same shape!")

#ifdef USE_CUDA
    if(x->devID >= 0 || y->devID >= 0){
        _CudaGELU(x, y);
        return;
    }
#endif

    DTYPE * ip = (DTYPE*)x->data;
    DTYPE * op = (DTYPE*)y->data;
    int n = x->GetSize();

    DTYPE sqrt_2_over_pi = sqrt(2.0 / M_PI);
    DTYPE scale = 0.5;

    for (int i = 0; i < n; i++) {
        DTYPE x = ip[i];
        DTYPE tanh_val = tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x));
        op[i] = scale * x * (1 + tanh_val);
    }
}

/*
rectify function y = max(0, x) (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input tensor
<< return - output tensor
*/
XTensor GELU(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Rectify function */
    _GELU(&x, &y);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_GELU);
    }

    return y;
}

void GELU(const XTensor &x, XTensor &y)
{
    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _Rectify function */
    _GELU(&x, &y);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_GELU);
    }
}

/*
backward computation

dE/dx = dE/dy * dy/dx

rectified: y = max(0, x)

or

rectified: y = 0     if x < 0
               x     otherwise

   and dy/ds = 0     if x < 0
               1     otherwise

>> y - output of the rectify function
>> x - input of the rectify function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _GELUBackward(XTensor * y, XTensor * x, 
                      XTensor * dedy, XTensor * dedx)
{
    CheckNTErrors(x != NULL, "The input tensor x must be not NULL!")

#ifdef USE_CUDA
    if(x->devID >= 0){
        _CudaGELUBackward(y, x, dedy, dedx);
        return;
    }
#endif

    DTYPE * dedyp = (DTYPE*)dedy->data;
    DTYPE * dedxp = (DTYPE*)dedx->data;
    DTYPE * ip = (DTYPE*)x->data;
    int size = x->unitNum;

    DTYPE sqrt_2_over_pi = sqrt(2.0 / M_PI);
    DTYPE a = 0.044715;
    DTYPE b = 0.134145;
    DTYPE c = 1.0;

    for (int i = 0; i < size; i++) {
        DTYPE x = ip[i];
        DTYPE tanh_val = tanh(sqrt_2_over_pi * (x + a * x * x * x));
        DTYPE derivative = 0.5 * (c + tanh_val) + 0.5 * x * (c - tanh_val * tanh_val) * sqrt_2_over_pi * (c + b * x * x);
        dedxp[i] = dedyp[i] * derivative;
    }
}

} // namespace nts(NiuTrans.Tensor)
