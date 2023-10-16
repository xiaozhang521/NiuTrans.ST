/* NiuTrans.S2T - an open-source speech to text system.
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
 * $Created by: Yuhao Zhang (yoohao.zhang) 2023-10-14
 */

#ifndef __S2T_GENERATOR_DATASET_H__
#define __S2T_GENERATOR_DATASET_H__

#include <string>
#include <fstream>
#include "../S2TVocab.h"
#include "../S2TDataset.h"


using namespace std;

/* the nmt namespace */
namespace s2t {
class S2TGeneratorDataset : public S2TDataSetBase {

public:

    /* whether append an empty line to the buffer */
    bool appendEmptyLine;

    /* the indices of empty lines */
    IntList emptyLines;

    /* the source vocabulary */
    S2TVocab srcVocab;

    /* the target vocabulary */
    S2TVocab tgtVocab;

    /* the input file stream */
    istream* ifp;

public:
    /* check if the buffer is empty */
    bool IsEmpty();

    /* initialization function */
    void Init(S2TConfig& myConfig, bool notUsed) override;

    /* load a sample from the buffer */
    TripleSample* LoadSample() override;

    /* transfrom a speech to a sequence */
    TripleSample* LoadSample(XTensor* s);

    /* transfrom a speech and a line to the sequence separately */
    TripleSample* LoadSample(XTensor* s, string line);

    /* load the samples into tensors from the buffer */
    bool GetBatchSimple(XList* inputs, XList* info) override;

    /* load the samples into the buffer (a list) */
    bool LoadBatchToBuf() override;

    /* constructor */
    S2TGeneratorDataset();

    /* de-constructor */
    ~S2TGeneratorDataset();
};
	

} /* end of s2t namespace */
#endif /* __S2T_GENERATOR_DATASET__ */