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

#include "../../XTensor.h"
#include "MakeSplitBlockIndex.h"
#include "MakeSplitBlockIndex.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
set target data block index for the data movement in split
>> blockIndex - block index
>> splitNum - number of splits
>> blockSplitSize - size of the splitted block
>> blockNum - number of data blocks
>> devID - device id
*/
void _MakeSplitBlockIndex(int * blockIndex, int splitNum, int blockSplitSize, int blockNum, int devID)
{
    if (devID >= 0) {
#ifdef USE_CUDA
        _CudaMakeSplitBlockIndex(devID, blockIndex, splitNum, blockSplitSize, blockNum);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        for (int i = 0; i < blockNum; i++) {
            int j = (i % splitNum) * blockSplitSize + i / splitNum;

            /* i = source block index, j = target block index */
            blockIndex[i] = j;
        }
    }
}
} // namespace nts(NiuTrans.Tensor)