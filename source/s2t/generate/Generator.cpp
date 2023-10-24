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

#include "Generator.h"
#include <iostream>

using namespace nts;
using namespace std;

namespace s2t
{
    /* constructor */
    Generator::Generator()
    {
        config = NULL;
        model = NULL;
        seacher = NULL;
        outputBuf = new XList;
    }

    /* de-constructor */
    Generator::~Generator()
    {
        if (config->inference.beamSize > 1)
            delete (BeamSearch*)seacher;
        else
            delete (GreedySearch*)seacher;
        delete outputBuf;
    }

    /* initialize the model */
    void Generator::Init(S2TConfig& myConfig, S2TModel& myModel)
    {
        cout << "----- Generator Init -----" << endl;
        model = &myModel;
        config = &myConfig;

        if (config->inference.beamSize > 1) {
            LOG("Inferencing with beam search (beam=%d, batchSize= %d sents | %d tokens, lenAlpha=%.2f, maxLenAlpha=%.2f) ",
                config->inference.beamSize, config->common.sBatchSize, config->common.wBatchSize,
                config->inference.lenAlpha, config->inference.maxLenAlpha);
            seacher = new BeamSearch();
            ((BeamSearch*)seacher)->Init(myConfig);
        }
        else if (config->inference.beamSize == 1) {
            LOG("Inferencing with greedy search (batchSize= %d sents | %d tokens, maxLenAlpha=%.2f)",
                config->common.sBatchSize, config->common.wBatchSize, config->inference.maxLenAlpha);
            seacher = new GreedySearch();
            ((GreedySearch*)seacher)->Init(myConfig);
        }
        else {
            CheckNTErrors(false, "Invalid beam size\n");
        }
        cout << "--- Generator Init End ---" << endl;
    }

    XTensor Generator::DecodingBatch(XTensor& batchEnc, XTensor& paddingEnc, IntList& indices)
    {
        // change single to batch
        bool isSingle = 0;
        if (batchEnc.order == 2) {
            isSingle = 1;
            int dim[3] = { 1, batchEnc.dimSize[0], batchEnc.dimSize[1] };
            batchEnc.Reshape(3, dim);
        }

        // begin decoding task
        int batchSize = batchEnc.GetDim(0);
        for (int i = 0; i < model->decoder->nlayer; ++i) {
            model->decoder->selfAttCache[i].miss = true;
            model->decoder->enDeAttCache[i].miss = true;
        }
        
        // encoder forward   *** should be in searcher, here test
        XTensor maskEnc;
        
        // encoder mask for test
        model->MakeS2TMaskEnc(paddingEnc, maskEnc);

        XTensor encoding;
        encoding = model->encoder->RunFastPreNorm(batchEnc, NULL);
        IntList** outputs = new IntList * [batchSize];

        encoding.Dump(stderr, "Encoder output is: ", 100);

        FILE* outputFile = fopen(config->inference.outputFN, "wb");
        if (outputFile)
            encoding.BinaryDump(outputFile);




        if (isSingle)
            return batchEnc;
        else {
            int dim[2] = { batchEnc.dimSize[1], batchEnc.dimSize[2] };
            batchEnc.Reshape(2, dim);
            return batchEnc;
        }
            
    }

    bool Generator::Generate()
    {
        batchLoader.Init(*config, false);
        
        /* inputs */
        XTensor batchEnc;
        XTensor paddingEnc;

        /* sentence information */
        XList info;
        XList inputs;
        int wordCount;
        IntList indices;
        inputs.Add(&batchEnc);
        inputs.Add(&paddingEnc);
        info.Add(&wordCount);
        info.Add(&indices);
        //TripleSample* longestSample = (TripleSample*)(batchLoader.buf->Get(0));
        //std::cout << longestSample->audioPath << endl;
        //longestSample->audioSeq->Dump();
        batchLoader.GetBatchSimple(&inputs, &info);
        batchEnc.Dump();
        //DecodingBatch(batchEnc, paddingEnc, indices);
        return true;
    }


    bool Generator::TestInference()         // not work for batch
    {
        
        // Pad audio 30s at right

        // extract fbank feature

        // Load test data from file for test
        XTensor test_audio_pad;
        InitTensor2D(&test_audio_pad, 3000, 80, X_FLOAT, config->common.devID);    // b * l * f
        FILE* audioFile = fopen(config->inference.inputFN, "rb");
        if (audioFile) {
            test_audio_pad.BinaryRead(audioFile);
        }

        XTensor paddingEnc;
        // init pad  for test
        InitTensor2D(&paddingEnc, 1, test_audio_pad.dimSize[0]/2, X_FLOAT, test_audio_pad.devID);
        paddingEnc = paddingEnc + 1;

        //DecodingBatch(test_audio_pad, paddingEnc);
            
        return 1;
    }

}