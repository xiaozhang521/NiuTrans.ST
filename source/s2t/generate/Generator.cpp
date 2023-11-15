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
            delete (S2TBeamSearch*)seacher;
        else
            delete (S2TGreedySearch*)seacher;
        seacher = nullptr;
        delete outputBuf;
    }

    /* initialize the model */
    void Generator::Init(S2TConfig& myConfig, S2TModel& myModel, OfflineFeatureTpl<FbankComputer>& myOft)
    {
        cout << "----- Generator Init -----" << endl;
        model = &myModel;
        config = &myConfig;
        oft = &myOft;

        if (config->inference.beamSize > 1) {
            LOG("Inferencing with beam search (beam=%d, batchSize= %d sents | %d tokens, lenAlpha=%.2f, maxLenAlpha=%.2f) ",
                config->inference.beamSize, config->common.sBatchSize, config->common.wBatchSize,
                config->inference.lenAlpha, config->inference.maxLenAlpha);
            seacher = new S2TBeamSearch();
            ((S2TBeamSearch*)seacher)->Init(myConfig);
        }
        else if (config->inference.beamSize == 1) {
            LOG("Inferencing with greedy search (batchSize= %d sents | %d tokens, maxLenAlpha=%.2f)",
                config->common.sBatchSize, config->common.wBatchSize, config->inference.maxLenAlpha);
            seacher = new S2TGreedySearch();
            ((S2TGreedySearch*)seacher)->Init(myConfig);
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
            batchEnc = Unsqueeze(batchEnc, 0, 1);
            paddingEnc = Unsqueeze(paddingEnc, 0, 1);
        }

        // begin decoding task
        int batchSize = batchEnc.GetDim(0);
        for (int i = 0; i < model->decoder->nlayer; ++i) {
            model->decoder->selfAttCache[i].miss = true;
            model->decoder->enDeAttCache[i].miss = true;
        }

        IntList** outputs = new IntList * [batchSize];
        for (int i = 0; i < batchSize; i++)
            outputs[i] = new IntList();

        /* greedy search */
        if (config->inference.beamSize == 1) {
            ((S2TGreedySearch*)seacher)->Search(model, batchEnc, paddingEnc, outputs);
        }
        else {
            XTensor score;
            ((S2TBeamSearch*)seacher)->Search(model, batchEnc, paddingEnc, outputs, score);
        }

        /*print output*/
        for (int i = 0; i < batchSize; i++) {
            cout << "batch:" << i << " output: ";
            for (int j = 0; j < outputs[i]->count; j++) {
                cout << outputs[i]->GetItem(j) << " ";
            }
            cout << endl;
        }

        string tokens = "";
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputs[i]->count; j++) {
                tokens += to_string(outputs[i]->GetItem(j));
                if (j < outputs[i]->count - 1)
                    tokens += " ";
            }
            if (i < batchSize)
                tokens += "\n";           
        }

        ofstream file(config->inference.outputFN, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
        }
        else {
            file << tokens;
            file.close();
        }
        

        if (isSingle) {
            /*TODO*/
            batchEnc = Squeeze(batchEnc);
        }
            
        return batchEnc;
            
    }

    bool Generator::Generate()
    {

        /* inputs */
        XTensor batchEnc;

        oft->Read();
        oft->ComputeFeatures(oft->Data().Data(), oft->Data().SampFreq(), 1.0, &batchEnc);
        batchEnc = Transpose(batchEnc, 0, 1);

        batchLoader.Init(*config, false);

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
        while (!batchLoader.IsEmpty()) {
            // batchLoader.GetBatchSimple(&inputs, &info);
            //batchEnc.Dump();
            //DecodingBatch(batchEnc, paddingEnc, indices);

            /*TODO wrong size*/
            //batchLoader.GetBatchSimple(&inputs, &info);
            // batchEnc.Dump(stderr, NULL, -1);

            //InitTensor3D(&batchEnc, 1, 3000, 80, X_FLOAT, config->common.devID);    // b * l * f
            //FILE* audioFile = fopen("../tools/data/batch.bin.using", "rb");
            //if (audioFile) {
            //    batchEnc.BinaryRead(audioFile);
            //}

            XTensor paddingEncForAudio;
            if (batchEnc.order == 3)
                InitTensor2D(&paddingEncForAudio, batchEnc.GetDim(0), int(batchEnc.GetDim(1) / 2), X_FLOAT, config->common.devID);
            else if (batchEnc.order == 2)
                InitTensor1D(&paddingEncForAudio, int(batchEnc.GetDim(0) / 2), X_FLOAT, config->common.devID);
            else
                CheckNTErrors(false, "Invalid batchEnc size\n");
            paddingEncForAudio = paddingEncForAudio * 0 + 1;

            DecodingBatch(batchEnc, paddingEncForAudio, indices);
        }

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