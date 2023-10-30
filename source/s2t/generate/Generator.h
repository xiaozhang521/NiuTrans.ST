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

 /*
 This class generate test speechs with a trained model.
 It will dump the result to the output file if specified, else the standard output.
 */

#ifndef __TRANSLATOR_S2T__
#define __TRANSLATOR_S2T__

#include "../S2TModel.h"
#include "../../nmt/translate/Searcher.h"
#include "S2TSearcher.h"
#include "S2TGeneratorDataSet.h"

 /* the s2t namespace */
namespace s2t
{

    class Generator
    {

    private:
        /* the translation model */
        S2TModel* model;

        /* for batching */
        S2TGeneratorDataset batchLoader;

        /* the searcher for translation */
        void* seacher;

        /* configuration of the NMT system */
        S2TConfig* config;

        /* output buffer */
        XList* outputBuf;

    private:
        /* translate a batch of sequences */
        XTensor DecodingBatch(XTensor& batchEnc, XTensor& paddingEnc, IntList& indices);



    public:
        /* constructor */
        Generator();

        /* de-constructor */
        ~Generator();

        ///* initialize the translator */
        void Init(S2TConfig& myConfig, S2TModel& myModel);

        /* the generate function */
        bool Generate();

        bool TestInference();

        ///* sort the outputs by the indices (in ascending order) */
        //void SortOutputs();

        ///* dump the translations to a file */
        //void DumpResToFile(const char* ofn);

        ///* dump the translations to stdout */
        //void DumpResToStdout();
    };

} /* end of the s2t namespace */

#endif /*  */