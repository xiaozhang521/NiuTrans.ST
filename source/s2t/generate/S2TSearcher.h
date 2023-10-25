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
  * $Created by: yuhao zhang(yoohao.zhang@gmail.com) 2023-09-22
  */

#ifndef __S2TSEARCHER_H__
#define __S2TSEARCHER_H__
#include "../S2TModel.h"
#include "../../nmt/translate/Searcher.h"

using namespace nmt;


namespace s2t {

    class S2TGreedySearch : public GreedySearch
    {
    private:
        /* max length of the generated sequence */
        int maxLen;

        /* batch size */
        int batchSize;

        /* array of the end symbols */
        int* endSymbols;

        /* number of the end symbols */
        int endSymbolNum;

        /* array of the start symbols */
        int* startSymbol;

        /* number of the start symbols */
        int startSymbolNum;

        /* scalar of the input sequence (for max number of search steps) */
        float scalarMaxLength;

    public:

        S2TGreedySearch();

        ~S2TGreedySearch();
        /* initialize the model */
        void Init(S2TConfig& config);

        void InitStartSymbol(S2TConfig& config);

        bool IsEnd(int token);

        /* search for the most promising states */
        void Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs);

        XTensor WhisperSuppress(XTensor& input);

        XTensor WhisperPredict(XTensor& tokens, XTensor& logits, XTensor& sumLogprobs);

    };

    class S2TBeamSearch : public BeamSearch
    {
    private:
        /* the alpha parameter controls the length preference */
        float alpha;

        /* predictor */
        Predictor predictor;

        /* max length of the generated sequence */
        int maxLen;

        /* beam size */
        int beamSize;

        /* batch size */
        int batchSize;

        /* we keep the final hypotheses in a heap for each sentence in the batch. */
        XHeap<MIN_HEAP, float>* fullHypos;

        /* array of the end symbols */
        int* endSymbols;

        /* number of the end symbols */
        int endSymbolNum;

        /* start symbol */
        int* startSymbols;

        /* number of the start symbols */
        int startSymbolNum;

        /* scalar of the input sequence (for max number of search steps) */
        float scalarMaxLength;

        /* indicate whether the early stop strategy is used */
        bool isEarlyStop;

        /* pids for alive states */
        IntList aliveStatePids;

        /* alive sentences */
        IntList aliveSentList;

        /* whether we need to reorder the states */
        bool needReorder;

    public:

        /*TODO*/
        /* constructor */
        S2TBeamSearch();

        /* de-constructor */
        ~S2TBeamSearch();

        /* initialize the model */
        void Init(S2TConfig& config);

        /* search for the most promising states */
        void Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** output, XTensor& score);
    };


} /* end of s2t namespace */



#endif