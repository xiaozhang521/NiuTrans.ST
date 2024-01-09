/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-06-22
*/

#include "../../XUtility.h"
#include "FlushToMem.h"
#include "FlushToMem.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
flush a list of XTensor to GPU memory
>> mList - list of the tensors
>> devID - target GPU id
>> GPUMem - memory pool for the GPU
*/
void CPUToGPUFlush(TensorList * mList, int devID, XMem * GPUMem)
{
#ifdef USE_CUDA
    CudaCPUToGPUFlush(mList, devID, GPUMem);
#endif
}

/* copy the data from GPU memory to CPU memory */
void GPUToCPUFlush(XTensor * tensor)
{
#ifdef USE_CUDA
    CudaGPUToCPUFlush(tensor);
#endif
}

} // namespace nts(NiuTrans.Tensor)