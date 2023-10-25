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

#include "S2TSearcher.h"
#include <iostream>

using namespace std;

namespace s2t {

	S2TGreedySearch::S2TGreedySearch()
	{
		maxLen = 0;
		batchSize = 0;
		endSymbolNum = 0;
		endSymbols = new int[32];
		startSymbolNum = 0;
		startSymbol = new int[32];
		scalarMaxLength = -1;
	}

	S2TGreedySearch::~S2TGreedySearch()
	{
		if (endSymbols != NULL)
			delete[] endSymbols;
		if (startSymbol != NULL)
			delete[] startSymbol;
	}

	void S2TGreedySearch::Init(S2TConfig& config)
	{
		maxLen = config.inference.maxLen;
		batchSize = config.common.sBatchSize;
		endSymbols[0] = config.model.eos;
		startSymbol[0] = config.model.sos;
		scalarMaxLength = config.inference.maxLenAlpha;

		if (endSymbols[0] >= 0)
			endSymbolNum = 1;
		if (startSymbol[0] >= 0)
			startSymbolNum = 1;

		InitStartSymbol(config);
	}

	void S2TGreedySearch::InitStartSymbol(S2TConfig& config)
	{
		CheckNTErrors(strcmp(config.whisperdec.language, "") != 0, "Invalid language tag");
		startSymbol[startSymbolNum++] = 50259; // en
		startSymbol[startSymbolNum++] = 50359;
		startSymbol[startSymbolNum++] = 50363;	// notimestamps
	}

	bool S2TGreedySearch::IsEnd(int token)
	{
		CheckNTErrors(endSymbolNum > 0, "No end symbol?");

		for (int i = 0; i < endSymbolNum; i++) {
			if (endSymbols[i] == token)
				return true;
		}

		return false;
	}

	XTensor S2TGreedySearch::WhisperSuppress(XTensor& input)
	{
		XTensor modify;
		InitTensor3D(&modify, 1, 1, 1, X_FLOAT, input.devID);
		modify = ScaleAndShift(modify, 0.0, -1e9);
		
		const int tokenNum = 88;
		// blank 220
		// <eot> 50257
		int suppressTokens[tokenNum] = { 1, 2, 7, 8, 9, 10, 14, 25, 26, 27,
								28, 29, 31, 58, 59, 60, 61, 62, 63, 90,
								91, 92, 93, 359, 503, 522, 542, 873, 893,
								902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246,
								3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273,
								9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157,
								14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520,
								26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
								47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362 };

		for (int i = 0; i < tokenNum; i++) {
			_SetDataIndexed(&input, &modify, input.order - 1, suppressTokens[i]);
		}

		return input;
	}

	XTensor S2TGreedySearch::WhisperPredict(XTensor& tokens, XTensor& logits, XTensor& sumLogprobs)
	{	
		// logits b * l * v
		/*TODO Temperature sets to 0.0*/
		XTensor nextToken, bestScore, logProbs;
		logits.Reshape(logits.dimSize[0], logits.dimSize[logits.order - 1]);	// b * v
		InitTensor2D(&nextToken, tokens.dimSize[0], 1, X_INT, tokens.devID);
		InitTensor2D(&bestScore, logits.dimSize[0], 1, X_FLOAT, logits.devID);
		TopK(logits, bestScore, nextToken, -1, 1);

		/*TODO calculate sumLogprobs*/
		logProbs = LogSoftmax(logits, -1);

		/*modify tokens to <eot> if it appeared before*/
		XTensor lastToken = SelectRange(tokens, tokens.order - 1, tokens.GetDim(tokens.order - 1) - 1, tokens.GetDim(tokens.order - 1));
		// lastToken.Dump(stderr, "Last Tokens: ", -1);
		for (int i = 0; i < lastToken.GetDim(0); i++) {
			if (IsEnd(lastToken.GetInt(i)))
				nextToken.Set2DInt(endSymbols[0], i, 0);
		}
		return nextToken;
	}

	void S2TGreedySearch::Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs)
	{
		cout << "----- S2TGreedySearch Search -----" << endl;
		// input.Dump(stderr, "input", 1);
		// padding.Dump(stderr, "padding", 1);
		
		XTensor maskEnc;
		XTensor encoding;
		batchSize = input.GetDim(0);

		/* encoder mask */
		model->MakeS2TMaskEnc(padding, maskEnc);

		/* make the encoding network */
		cout << "----- Encoding -----" << endl;
		if (model->config->model.encPreLN)
			encoding = model->encoder->RunFastPreNorm(input, &maskEnc);
		// else
			/*TODO*/
			// encoding = model->encoder->RunFastPostNorm(input, &maskEnc);

		// encoding.Dump(stderr, "Encoder output is: ", 100);
		// FILE* encOutput = fopen("../tools/data/encOutput.bin", "wb");
		// encoding.BinaryDump(encOutput);

		cout << "--- Encoding End ---" << endl;

		/* max output-length = scalar * source-length */
		int lengthLimit = MIN(int(float(input.GetDim(-2)) * scalarMaxLength), maxLen);
		CheckNTErrors(lengthLimit > 0, "Invalid maximum output length");
		cout << "lengthLimit: " << lengthLimit << endl;

		/* the first token */
		XTensor inputDec;
		InitTensor1D(&inputDec, startSymbolNum, X_INT, input.devID);
		inputDec.SetData(startSymbol, startSymbolNum);
		inputDec = Unsqueeze(inputDec, 0, batchSize);


		/* initialize the finished flags */
		int* finishedFlags = new int[batchSize];
		for (int i = 0; i < batchSize; i++)
			finishedFlags[i] = 0;

		XTensor prob;
		XTensor maskDec;
		XTensor decoding;
		XTensor indexCPU;
		XTensor bestScore;

		InitTensor2D(&indexCPU, batchSize, 1, inputDec.dataType, -1);
		InitTensor2D(&bestScore, batchSize, 1, encoding.dataType, encoding.devID);

		// FILE* audioFeature = fopen("../tools/data/audio_features.bin", "rb");
		// encoding.BinaryRead(audioFeature);

		// encoding.Dump(stderr, "Decoder input(Encoder output): ", 20);
		// inputDec.Dump(stderr, "Decoder input(Tokens): ", -1);
		int initTokenLen = inputDec.GetDim(-1);

		/* decoder mask */
		maskDec = model->MakeS2TTriMaskDecInference(batchSize, inputDec.GetDim(-1));
		// maskDec.Dump(stderr, "maskEncDec: ", -1);

		model->decoder->embedder->scale = FALSE;
		
		for (int l = 0; l < lengthLimit; l++) {

			// cout << "----- Decoding -----" << l << endl;

			int nstep = l;
			if (l > 0)
				nstep += (initTokenLen - 1);
			/* make the decoding network */
			if (model->config->model.decPreLN)
				if ( l == 0 )
					decoding = model->decoder->RunFastPreNorm(inputDec, encoding, &maskDec, NULL, nstep);
				else
					decoding = model->decoder->RunFastPreNorm(inputDec, encoding, NULL, NULL, nstep);

				// FILE* decOutput = fopen("../tools/data/decOutput.bin", "wb");
				// decoding.BinaryDump(decOutput);
				// decoding.Dump(stderr, "decOutput: ", 10);
			//else
				/*TODO*/
				//decoding = model->decoder->RunFastPostNorm(inputDec, encoding, &maskEncDec, l);

			/* generate the output probabilities */
			XTensor logits;
			logits = MMul(decoding, X_NOTRANS, *model->outputLayer->w, X_TRANS);

			// FILE* probOutput = fopen("../tools/data/probOutput.bin", "wb");
			// logits.BinaryDump(probOutput);
			// logits.Dump(stderr, "probOutput: ", 10);

			/*calculate prob of no_speech*/
			if (l == 0) {
				// no speech token 50362 TODO

			}

			/*only consider the last token*/
			logits = SelectRange(logits, 1, logits.GetDim(1) - 1, logits.GetDim(1));
			// logits.Dump(stderr, "logits: ", 10);

			/*apply the logit filters*/
			XTensor logitsFilted;
			logitsFilted = WhisperSuppress(logits);

			/*calculate next token*/
			XTensor sumLogprobs, nextToken;
			InitTensor1D(&sumLogprobs, batchSize, X_FLOAT, logitsFilted.devID);
			nextToken = WhisperPredict(inputDec, logitsFilted, sumLogprobs);
			// nextToken.Dump(stderr, "New inputDec: ", -1);

			/* save the predictions */
			CopyValues(nextToken, indexCPU);

			for (int i = 0; i < batchSize; i++) {
				if (IsEnd(indexCPU.GetInt(i)))
					finishedFlags[i] = 1;
				else if (finishedFlags[i] != 1)
					(outputs[i])->Add(indexCPU.GetInt(i));
			}
			
			/*next loop*/
			inputDec = nextToken;

			// cout << "--- Decoding End ---" << l << endl;

			int finishedSentNum = 0;
			for (int i = 0; i < batchSize; i++)
				finishedSentNum += finishedFlags[i];
			if (finishedSentNum == batchSize) {
				l = lengthLimit;
				break;
			}
		}

		/*print output*/
		for (int i = 0; i < batchSize; i++) {
			cout << "batch:" << i << "output: ";
			for (int j = 0; j < outputs[i]->count; j++) {
				cout << outputs[i]->GetItem(j) << " ";
			}
			cout << endl;
		}

		cout << "--- S2TGreedySearch Search End ---" << endl;
	}

	

}