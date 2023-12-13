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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "FFN.h"
#include "Embedding.h"
#include "../Config.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/function/FHeader.h"
#include "../../utils/timer.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void FFN::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
FFN::FFN()
{
    dropoutP = 0.0F;
    inSize = -1;
    outSize = -1;
    hSize = -1;
    devID = -1;
    actFunType = ReLUFun;
    isTraining = false;
}

/* de-constructor */
FFN::~FFN()
{
}

/*
initialize the model
>> config - configurations of the model
>> isEnc - indicates wether it is a encoder module
*/
void FFN::InitModel(NMTConfig& config, bool isEnc)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    dropoutP = config.model.ffnDropout;
    inSize = isEnc ? config.model.encEmbDim : config.model.decEmbDim;
    outSize = isEnc ? config.model.encEmbDim : config.model.decEmbDim;
    hSize = isEnc ? config.model.encFFNHiddenDim : config.model.decFFNHiddenDim;
    actFunType = config.model.fnnActFunType;

    InitTensor2D(&w1, inSize, hSize, X_FLOAT, devID);
    InitTensor1D(&b1, hSize, X_FLOAT, devID);

    InitTensor2D(&w2, hSize, outSize, X_FLOAT, devID);
    InitTensor1D(&b2, outSize, X_FLOAT, devID);

    if (isTraining) {
        _SetDataFanInOut(&w1);
        _SetDataFanInOut(&w2);

        b1.SetZeroAll();
        b2.SetZeroAll();
    }
}

/*
make the network
y = max(0, x * w1 + b1) * w2 + b2
>> input - the input tensor
>> return - the output tensor
*/
XTensor FFN::Make(XTensor& input)
{
    util::timer_c timer;
    timer.m_start_timer();          

    XTensor t1;

    /* t1 = max(0, x * w1 + b1) */
    if(actFunType == ReLUFun)
        t1 = Rectify(MulAndShift(input, w1, b1));
    else if (actFunType == GELUFun)
        t1 = GELU(MulAndShift(input, w1, b1));
    if (isTraining && dropoutP > 0)
        t1 = Dropout(t1, dropoutP, /*inplace=*/true);

    /* result = t1 * w2 + b2 */
    t1=MulAndShift(t1, w2, b2);
    timer.m_end_timer();
    time_ffn += timer.m_get_time_diff_msec();
    return t1;
}

} /* end of the nmt namespace */