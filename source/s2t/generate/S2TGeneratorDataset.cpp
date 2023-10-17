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

#include <iostream>
#include <algorithm>
#include "S2TGeneratorDataset.h"
#include "../../niutensor/tensor/XTensor.h"
#include <unordered_map>

using namespace nts;

/* the S2T namespace */
namespace s2t {

/* transfrom a speech to a sequence */
TripleSample* S2TGeneratorDataset::LoadSample(XTensor* s)
{
    TripleSample* sample = new TripleSample(s);
    return sample;
}


/* transfrom a speech and a line to the sequence separately */
TripleSample* S2TGeneratorDataset::LoadSample(XTensor* s, string line)
{
    const string delimiter = " ";

    /* load tokens and transform them to ids */
    vector<string> srcTokens = SplitString(line, delimiter,
        config->model.maxSrcLen - 1);

    IntList* srcSeq = new IntList(int(srcTokens.size()));
    TripleSample* sample = new TripleSample(s, srcSeq);

    for (const string& token : srcTokens) {
        if (srcVocab.token2id.find(token) == srcVocab.token2id.end())
            srcSeq->Add(srcVocab.unkID);
        else
            srcSeq->Add(srcVocab.token2id.at(token));
    }

    /* the sequence should ends with EOS */
    if (srcSeq->Get(-1) != srcVocab.eosID)
        srcSeq->Add(srcVocab.eosID);

    return sample;
}

/* this is a place-holder function to avoid errors */
TripleSample* S2TGeneratorDataset::LoadSample()
{
    return nullptr;
}

/*
read data from a file to the buffer
*/
bool S2TGeneratorDataset::LoadBatchToBuf()
{
    int id = 0;
    ClearBuf();
    emptyLines.Clear();

    string line;
    getline(*ifp, line);
    /* get the tag index of each column*/
    unordered_map<string, int> tagsMap;
    int start = 0;
    int index = line.find('\t');
    int cot = 0;
    string tags[5] = { "audio","frames","tgt_text","src_text","speaker" };
    while (index != string::npos)
    {
        string tmpTag = line.substr(start, index - start);
        int tagIndex = find(tags, tags + 5, tmpTag) - tags;
        if (tagIndex < 5)
            tagsMap[tmpTag] = cot;
        /* record the index of neccesary line */
        cot += 1;
        start = index + 1;
        index = line.find('\t',start);
    }
    
    string tmpTag = line.substr(start, index - start);
    int tagIndex = find(tags, tags + 5, tmpTag) - tags;
    if (tagIndex < 5)
        tagsMap[tmpTag] = cot;
    while (getline(*ifp, line) && id < config->common.bufSize) {

        /* handle empty lines */
        if (line.size() > 0) {
            vector<string> tmpStrings;
            start = 0;
            index = line.find('\t', start);
            while (index != string::npos)
            {
                tmpStrings.push_back(line.substr(start, index - start));
                start = index + 1;
                index = line.find('\t', start);
            }
            tmpStrings.push_back(line.substr(start, index - start));

            XTensor inputAudio;
            InitTensor2D(&inputAudio, (int)stol(tmpStrings[tagsMap["frames"]]), config->s2tmodel.fbank, X_FLOAT, config->common.devID);
            FILE* audioFile = fopen(tmpStrings[tagsMap["audio"]].data(), "rb");
            if (audioFile)
                inputAudio.BinaryRead(audioFile);
            TripleSample* sequence = LoadSample(&inputAudio);
            sequence->index = id;
            buf->Add(sequence);
        }
        else {
            emptyLines.Add(id);
        }

        id++;
    }

    /* hacky code to solve the issue with fp16 */
    /* TODO!!! update the empty line */
    /*appendEmptyLine = false;
    if (id > 0 && id % 2 != 0) {
        line = "EMPTY";
        XTensor inputAudio;
        InitTensor2D(&inputAudio, 1, 1, X_FLOAT, config->common.devID);
        TripleSample* sequence = LoadSample(&inputAudio);
        sequence->index = id++;
        buf->Add(sequence);
        appendEmptyLine = true;
    }*/

    SortBySrcLengthDescending();
    XPRINT1(0, stderr, "[INFO] loaded %d sentences\n", appendEmptyLine ? id - 1 : id);

    return true;
}

/* constructor */
S2TGeneratorDataset::S2TGeneratorDataset()
{
    ifp = NULL;
    appendEmptyLine = false;
}

/*
load a batch of sequences from the buffer to the host for translating
>> inputs - a list of input tensors (batchEnc and paddingEnc)
   batchEnc - a tensor to store the batch of input
   paddingEnc - a tensor to store the batch of paddings
>> info - the total length and indices of sequences
*/
bool S2TGeneratorDataset::GetBatchSimple(XList* inputs, XList* info)
{
    int realBatchSize = 1;

    /* get the maximum sequence length in a mini-batch */
    TripleSample* longestsample = (TripleSample*)(buf->Get(bufIdx));
    int maxLen = longestsample->fLen;

    /* we choose the max-token strategy to maximize the throughput */
    while (realBatchSize * maxLen * config->inference.beamSize < config->common.wBatchSize
        && realBatchSize < config->common.sBatchSize) {
        realBatchSize++;
    }

    realBatchSize = MIN(realBatchSize, config->common.sBatchSize);

    /* make sure the batch size is valid */
    realBatchSize = MIN(int(buf->Size()) - bufIdx, realBatchSize);
    realBatchSize = MAX(2 * (realBatchSize / 2), realBatchSize % 2);

    CheckNTErrors(maxLen != 0, "Invalid length");

    int* batchValues = new int[realBatchSize * maxLen];
    float* paddingValues = new float[realBatchSize * maxLen];

    for (int i = 0; i < realBatchSize * maxLen; i++) {
        batchValues[i] = srcVocab.padID;
        paddingValues[i] = 1.0F;
    }

    int* totalLength = (int*)(info->Get(0));
    IntList* indices = (IntList*)(info->Get(1));
    *totalLength = 0;
    indices->Clear();

    /* right padding */
    /* TODO!!! Check the length of audio */
    /*int curSrc = 0;
    for (int i = 0; i < realBatchSize; ++i) {
        TripleSample* sequence = (TripleSample*)(buf->Get(bufIdx + i));
        IntList* src = sequence->srcSeq;
        indices->Add(sequence->index);
        *totalLength += src->Size();

        curSrc = maxLen * i;
        memcpy(&(batchValues[curSrc]), src->items, sizeof(int) * src->Size());
        curSrc += src->Size();

        while (curSrc < maxLen * (i + 1))
            paddingValues[curSrc++] = 0.0F;
    }*/

    bufIdx += realBatchSize;

    XTensor* batchEnc = (XTensor*)(inputs->Get(0));
    XTensor* paddingEnc = (XTensor*)(inputs->Get(1));
    InitTensor2D(batchEnc, realBatchSize, maxLen, X_INT, config->common.devID);
    InitTensor2D(paddingEnc, realBatchSize, maxLen, config->common.useFP16 ? X_FLOAT : X_FLOAT, config->common.devID);
    batchEnc->SetData(batchValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingValues, paddingEnc->unitNum);

    delete[] batchValues;
    delete[] paddingValues;

    return true;
}

/*
constructor
>> myConfig - configuration of the NMT system
>> notUsed - as it is
*/
void S2TGeneratorDataset::Init(S2TConfig& myConfig, bool notUsed)
{
    config = &myConfig;
    
    /* load the source and target vocabulary */
    tgtVocab.Load(config->common.tgtVocabFN);

    /* share the source and target vocabulary */
    if (strcmp(config->common.srcVocabFN, "") != 0)
    {
        if (strcmp(config->common.srcVocabFN, config->common.tgtVocabFN) == 0)
            srcVocab.CopyFrom(tgtVocab);
        else
            srcVocab.Load(config->common.srcVocabFN);
        srcVocab.SetSpecialID(config->model.sos, config->model.eos,
            config->model.pad, config->model.unk);
    }
    
    tgtVocab.SetSpecialID(config->model.sos, config->model.eos,
        config->model.pad, config->model.unk);

    /* translate the content in a file */
    if (strcmp(config->inference.inputFN, "") != 0) {
        ifp = new ifstream(config->inference.inputFN);
        CheckNTErrors(ifp, "Failed to open the input file");
    }
    /* translate the content in stdin */
    else
        ifp = &cin;

    LoadBatchToBuf();
}

/* check if the buffer is empty */
bool S2TGeneratorDataset::IsEmpty() {
    if (bufIdx < buf->Size())
        return false;
    return true;
}

/* de-constructor */
S2TGeneratorDataset::~S2TGeneratorDataset()
{
    if (ifp != NULL && strcmp(config->inference.inputFN, "") != 0) {
        ((ifstream*)(ifp))->close();
        delete ifp;
    }
}

} /* end of the s2t namespace */