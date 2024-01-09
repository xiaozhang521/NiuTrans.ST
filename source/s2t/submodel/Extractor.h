/* NiuTrans.S2T - an open-source speech-to-text system.
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
 * $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-09-19
 */

#ifndef __EXTRACTOR_H__
#define __EXTRACTOR_H__

#include "../S2TConfig.h"
#include "../../niutensor/tensor/XTensor.h"

using namespace nts;

/* the s2t namespace */
namespace s2t
{

/* a fnn: y = max(0, x * w1 + b1) * w2 + b2 */
class Extractor
{
public:
    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* size of input vector */
    int inSize;

    /* size of output vector */
    int hSize;

    /* number of convolution layer */
    int nConv;

    /* kernel sizes of convolution layer */
    vector<int> convKernels;

    /* stride sizes of convolution layer */
    vector<int> convStrides;

    /* matrix of kernel tensor */
    XTensor* kernels;

    /* matrix of convolution bias tensor */
    XTensor* biases;

    /* dropout probability */
    DTYPE dropoutP;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    Extractor();

    /* de-constructor */
    ~Extractor();

    /* initialize the model */
    void InitModel(S2TConfig& config);

    /* make the network */
    XTensor Make(XTensor& input);
};

} /* end of the s2t namespace */

#endif /* __EXTRACTOR_H__ */