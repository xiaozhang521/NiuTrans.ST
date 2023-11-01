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
		startSymbols = new int[32];
		suppressSymbolNum = 0;
		suppressSymbols = new int[100];
		scalarMaxLength = -1;
	}

	S2TGreedySearch::~S2TGreedySearch()
	{
		if (endSymbols != NULL)
			delete[] endSymbols;
		if (startSymbols != NULL)
			delete[] startSymbols;
		if (suppressSymbols != NULL)
			delete[] suppressSymbols;
	}

	void S2TGreedySearch::Init(S2TConfig& config)
	{
		maxLen = config.inference.maxLen;
		batchSize = config.common.sBatchSize;
		endSymbols[0] = config.model.eos;
		startSymbols[0] = config.model.sos;
		scalarMaxLength = config.inference.maxLenAlpha;

		if (endSymbols[0] >= 0)
			endSymbolNum = 1;
		if (startSymbols[0] >= 0)
			startSymbolNum = 1;

		InitStartSymbols(config);

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

		InitSuppressSymbols(config, suppressTokens, tokenNum);
	}

	void S2TGreedySearch::InitStartSymbols(S2TConfig& config)
	{
		CheckNTErrors(strcmp(config.whisperdec.language, "") != 0, "Invalid language tag");
		startSymbols[startSymbolNum++] = 50259; // en 50259
		startSymbols[startSymbolNum++] = 50359;
		startSymbols[startSymbolNum++] = 50363; // notimestamps
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

	void S2TGreedySearch::InitSuppressSymbols(S2TConfig& config, int* tokens, const int num)
	{
		if ( num > 0 )
		{
			/*init suppress symbol from tokens*/ 
			CheckNTErrors(num <= 100, "Invalid suppress token length ( should less than 100 )");
			suppressSymbolNum = num;
			// blank 220
			// <eot> 50257
			for (int i = 0; i < suppressSymbolNum; i++) {
				suppressSymbols[i] = tokens[i];
			}
		}
		else {
			/*init suppress symbol from config*/
			/*TODO*/

		}
		
	}

	XTensor S2TGreedySearch::Suppress(XTensor& input)
	{
		XTensor modify;
		InitTensor3D(&modify, input.GetDim(0), 1, 1, X_FLOAT, input.devID);
		modify = ScaleAndShift(modify, 0.0, -1e9);

		if (suppressSymbolNum <= 0)
			return input;

		for (int i = 0; i < suppressSymbolNum; i++) {
			_SetDataIndexed(&input, &modify, input.order - 1, suppressSymbols[i]);
		}

		return input;
	}

	XTensor S2TGreedySearch::Predict(XTensor& tokens, XTensor& logits, XTensor* sumLogprobs)
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
		inputDec.SetData(startSymbols, startSymbolNum);
		inputDec = Unsqueeze(inputDec, 0, batchSize);


		/* initialize the finished flags */
		int* finishedFlags = new int[batchSize];
		for (int i = 0; i < batchSize; i++)
			finishedFlags[i] = 0;

		XTensor prob, nextToken;
		XTensor maskDec;
		XTensor decoding;
		XTensor indexCPU;
		XTensor bestScore;

		InitTensor2D(&indexCPU, batchSize, 1, inputDec.dataType, -1);
		InitTensor2D(&bestScore, batchSize, 1, encoding.dataType, encoding.devID);

		// FILE* audioFeature = fopen("../tools/data/audio_features.bin", "rb");
		// encoding.BinaryRead(audioFeature);

		// encoding.Dump(stderr, "Decoder input(Encoder output): ", 20);
		inputDec.Dump(stderr, "Decoder input(Tokens): ", -1);

		int initTokenLen = inputDec.GetDim(-1);

		/* decoder mask */
		maskDec = model->MakeS2TTriMaskDecInference(batchSize, inputDec.GetDim(-1));
		// maskDec.Dump(stderr, "maskEncDec: ", -1);

		model->decoder->embedder->scale = false;
		
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
			/*TODO*/
			bool outputSoftmax = false;
			if (outputSoftmax)
				prob = model->outputLayer->Make(decoding, false);
			else
				prob = MMul(decoding, X_NOTRANS, *model->outputLayer->w, X_TRANS);

			// FILE* probOutput = fopen("../tools/data/probOutput.bin", "wb");
			// logits.BinaryDump(probOutput);
			// logits.Dump(stderr, "probOutput: ", 10);

			/*calculate prob of no_speech (whisper) */
			if (l == 0) {
				// no speech token 50362 TODO

			}

			/*only consider the last token*/
			prob = SelectRange(prob, 1, prob.GetDim(1) - 1, prob.GetDim(1));
			// logits.Dump(stderr, "logits: ", 10);

			/*apply the logit filters*/
			XTensor probFilted;
			probFilted = Suppress(prob);

			/*calculate next token*/
			nextToken = Predict(inputDec, probFilted);
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

		cout << "--- S2TGreedySearch Search End ---" << endl;
	}

	S2TBeamSearch::S2TBeamSearch()
	{
		alpha = 0;
		maxLen = 0;
		beamSize = 0;
		batchSize = 0;
		endSymbolNum = 0;
		startSymbolNum = 0;
		suppressSymbolNum = 0;
		fullHypos = NULL;
		endSymbols = new int[32];
		startSymbols = new int[32];
		suppressSymbols = new int[100];
		isEarlyStop = false;
		needReorder = false;
		scalarMaxLength = 0.0F;
	}

	/* de-constructor */
	S2TBeamSearch::~S2TBeamSearch()
	{
		if (fullHypos != NULL)
			delete[] fullHypos;
		if (endSymbols != NULL)
			delete[] endSymbols;
		if (startSymbols != NULL)
			delete[] startSymbols;
		if (suppressSymbols != NULL)
			delete[] suppressSymbols;
	}

	void S2TBeamSearch::Init(S2TConfig& config)
	{
		maxLen = config.inference.maxLen;
		beamSize = config.inference.beamSize;
		batchSize = config.common.sBatchSize;
		alpha = config.inference.lenAlpha;
		endSymbols[0] = config.model.eos;
		startSymbols[0] = config.model.sos;
		scalarMaxLength = config.inference.maxLenAlpha;

		if (endSymbols[0] >= 0)
			endSymbolNum = 1;
		if (startSymbols[0] >= 0)
			startSymbolNum = 1;

		InitStartSymbols(config);

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

		InitSuppressSymbols(config, suppressTokens, tokenNum);
		// BeamSearch::Init((NMTConfig)(config));
		BeamSearch::Init(endSymbols, endSymbolNum, startSymbols[0], maxLen, beamSize, batchSize, alpha, scalarMaxLength);
	}

	void S2TBeamSearch::InitStartSymbols(S2TConfig& config)
	{
		CheckNTErrors(strcmp(config.whisperdec.language, "") != 0, "Invalid language tag");
		startSymbols[startSymbolNum++] = 50259; // en 50259
		startSymbols[startSymbolNum++] = 50359;
		startSymbols[startSymbolNum++] = 50363; // notimestamps
	}

	void S2TBeamSearch::InitSuppressSymbols(S2TConfig& config, int* tokens, const int num)
	{
		if (num > 0)
		{
			/*init suppress symbol from tokens*/
			CheckNTErrors(num <= 100, "Invalid suppress token length ( should less than 100 )");
			suppressSymbolNum = num;
			// blank 220
			// <eot> 50257
			for (int i = 0; i < suppressSymbolNum; i++) {
				suppressSymbols[i] = tokens[i];
			}
		}
		else {
			/*init suppress symbol from config*/
			/*TODO*/

		}

	}

	/*
	prepare for search
	>> batchSize - size of the batch
	>> beamSize - size of the beam
	*/
	void S2TBeamSearch::Prepare(int myBatchSize, int myBeamSize)
	{
		batchSize = myBatchSize;
		beamSize = myBeamSize;
		needReorder = false;

		/* prepare for the heap of hypotheses */
		if (fullHypos != NULL)
			delete[] fullHypos;

		fullHypos = new XHeap<MIN_HEAP, float>[batchSize];

		for (int i = 0; i < batchSize; i++)
			fullHypos[i].Init(beamSize);

		/* prepare for the indices of alive states */
		aliveStatePids.Clear();
		aliveSentList.Clear();
		for (int i = 0; i < batchSize; i++) {
			aliveStatePids.Add(i);
			aliveSentList.Add(i);
		}
	}

	/*
	collect hypotheses with ending symbols. Given a beam of hypotheses,
	we remove the finished hypotheses and keep them in a heap.
	>> beam  - the beam that keeps a number of states
	*/
	void S2TBeamSearch::Collect(StateBundle* beam)
	{
		State* states = beam->states;

		for (int i = 0; i < beam->stateNum; i++) {
			State& state = states[i];

			CheckNTErrors(state.pid >= 0 && state.pid < batchSize,
				"Invalid sample id!");

			/* check if this is the first end symbol. It is false
			   if there have been end symbols in previously generated words. */
			bool isCompleted = state.isCompleted &&
				(state.last == NULL || !state.last->isCompleted);

			/* we push the hypothesis into the heap when it is completed */
			if ((state.isEnd || state.isCompleted)) {
				fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
			}
		}
	}

	/*
	fill the hypothesis heap with incomplete hypotheses
	>> beam  - the beam that keeps a number of states (final)
	*/
	void S2TBeamSearch::FillHeap(StateBundle* beam)
	{
		State* states = beam->states;

		for (int i = 0; i < beam->stateNum / beamSize; i++) {
			for (int j = 0; j < beamSize; j++) {
				State& state = states[i * beamSize + j];

				/* we push the incomplete hypothesis into the heap */
				if (fullHypos[state.pid].Count() == 0) {
					fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
				}
				else {
					auto node = fullHypos[state.pid].Top();
					float score = node.value;
					if (score < state.modelScore)
						fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
				}
			}
		}
	}

	/*
	save the output sequences in a tensor
	>> output - output sequences (for return)
	>> score - score of thes sequences
	*/
	void S2TBeamSearch::Dump(IntList** output, XTensor* score)
	{
		int dims[3] = { batchSize, 1 };

		InitTensor(score, 2, dims, X_FLOAT);
		score->SetZeroAll();

		/* heap for an input sentence in the batch */
		for (int h = 0; h < batchSize; h++) {
			IntList* tgt = output[h];
			XHeap<MIN_HEAP, float>& heap = fullHypos[h];
			int c = heap.Count();

			float bestScore = -1e9F;
			State* state = NULL;
			for (int i = 0; i < c; i++) {
				auto node = heap.Pop();
				State* s = (State*)node.index;
				if (i == 0 || bestScore < node.value) {
					state = s;
					bestScore = node.value;
				}
			}

			int count = 0;
			bool isCompleted = true;

			/* we track the state from the end to the beginning */
			while (state != NULL) {
				if (!state->isCompleted)
					isCompleted = false;
				if (!isCompleted) {
					tgt->Add(state->prediction);
				}
				state = state->last;
			}
			tgt->Reverse();

			score->Set2D(bestScore, h, 0);
		}
	}

	/*
	search for the most promising states
	>> model - the transformer model
	>> input - input of the model
	>> padding - padding of the input
	>> outputs - outputs that represent the sequences as rows
	>> score - score of the sequences
	*/
	void S2TBeamSearch::Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs, XTensor& score)
	{
		cout << "----- S2TBeamSearch Search -----" << endl;
		
		S2TPredictor predictor;
		XTensor maskEnc;
		XTensor encoding;
		XTensor encodingBeam;
		XTensor inputBeam;
		XTensor paddingBeam;

		CheckNTErrors(endSymbolNum > 0, "The search class is not initialized!");
		CheckNTErrors(startSymbolNum > 0, "The search class is not initialized!");

		Prepare(input.GetDim(0), beamSize);

		/* encoder mask */
		model->MakeS2TMaskEnc(padding, maskEnc);

		/* make the encoding network */
		cout << "----- Encoding -----" << endl;
		if (model->config->model.encPreLN)
			encoding = model->encoder->RunFastPreNorm(input, &maskEnc);
		// else
			/*TODO*/
			// encoding = model->encoder->RunFastPostNorm(input, &maskEnc);
		cout << "--- Encoding End ---" << endl;

		encodingBeam = Unsqueeze(encoding, encoding.order - 2, beamSize);
		inputBeam = Unsqueeze(input, input.order - 2, beamSize);
		paddingBeam = Unsqueeze(padding, padding.order - 1, beamSize);

		encodingBeam.ReshapeMerged(encodingBeam.order - 4);
		inputBeam.ReshapeMerged(inputBeam.order - 4);
		paddingBeam.ReshapeMerged(paddingBeam.order - 3);

		/* max output-length = scalar * source-length */
		int lengthLimit = MIN(int(float(input.GetDim(-2)) * scalarMaxLength), maxLen);
		CheckNTErrors(lengthLimit > 0, "Invalid maximum output length");
		cout << "lengthLimit: " << lengthLimit << endl;

		StateBundle* states = new StateBundle[lengthLimit + 1];
		StateBundle* first = states;
		StateBundle* cur = NULL;
		StateBundle* next = NULL;

		/* create the first state */
		predictor.Init(endSymbols, endSymbolNum, startSymbols, startSymbolNum, suppressSymbols, suppressSymbolNum);
		predictor.Create(&input, beamSize, first);
		
		first->isStart = true;

		XTensor aliveState;
		InitTensor1D(&aliveState, batchSize * beamSize, X_INT, input.devID);
		SetAscendingOrder(aliveState, 0);

		XTensor reorderState;
		InitTensor1D(&reorderState, batchSize * beamSize, X_INT, input.devID);
		SetAscendingOrder(reorderState, 0);

		model->decoder->embedder->scale = false;

		/* generate the sequence from left to right */
		// lengthLimit = 1;
		for (int l = 0; l < lengthLimit; l++) {

			int nstep = l;
			if (l > 0)
				nstep += (startSymbolNum - 1);
			
			if (beamSize > 1) {
				inputBeam = AutoGather(inputBeam, reorderState);
				paddingBeam = AutoGather(paddingBeam, reorderState);
				encodingBeam = AutoGather(encodingBeam, reorderState);
			}

			cur = states + l;
			next = states + l + 1;

			/* read the current state */
			predictor.Read(model, cur);

			/* predict the next state */
			predictor.Predict(next, aliveState, encodingBeam, inputBeam,
				paddingBeam, batchSize * beamSize, l == 0, reorderState, needReorder, nstep);

			/* compute the model score (given the prediction probability) */
			Score(cur, next);

			next->prob.enableGrad = false;
			next->probPath.enableGrad = false;
			next->modelScore.enableGrad = false;

			/* beam pruning */
			Generate(cur, next);

			/* expand the search graph */
			Expand(cur, next, reorderState);

			/* push complete hypotheses into the heap */
			Collect(next);

			next->prediction.Dump(stderr, "\prediction: ", -1);
			next->probPath.Dump(stderr, "probPath: ", -1);

			/* stop searching when all hypotheses are completed */
			if (IsAllCompleted(next)) {
				break;
			}

			/* remove finished sentences */
			//RemoveFinishedStates(next, encodingBeam, inputBeam, paddingBeam, aliveState);
		}

		/* fill the heap with incomplete hypotheses if necessary */
		FillHeap(next);

		Dump(outputs, &score);

		delete[] states;

		cout << "--- S2TBeamSearch Search End ---" << endl;
	}

	S2TPredictor::S2TPredictor()
	{
		m = NULL;
		s = NULL;
		endSymbols = NULL;
		endSymbolNum = 0;
		startSymbols = NULL;
		startSymbolNum = 0;
		suppressSymbols = NULL;
		suppressSymbolNum = 0;
	}

	S2TPredictor::~S2TPredictor()
	{
		m = NULL;
		s = NULL;
		endSymbols = NULL;
		startSymbols = NULL;
		suppressSymbols = NULL;
	}

	void S2TPredictor::Init(int* endS, int endN, int* startS, int startN, int* suppS, int suppN)
	{
		endSymbols = endS;
		endSymbolNum = endN;
		startSymbols = startS;
		startSymbolNum = startN;
		suppressSymbols = suppS;
		suppressSymbolNum = suppN;
	}

	/*
	create an initial state
	>> model - the  model
	>> top - the top-most layer of the network
	>> input - input of the network
	>> beamSize - beam size
	>> state - the state to be initialized
	*/
	void S2TPredictor::Create(const XTensor* input, int beamSize, StateBundle* state)
	{
		/*TODO*/
		int dims[MAX_TENSOR_DIM_NUM];
		dims[0] = input->dimSize[0];
		dims[1] = beamSize;

		InitTensor(&state->probPath, input->order-1, dims, X_FLOAT, input->devID);
		InitTensor(&state->endMark, input->order-1, dims, X_INT, input->devID);

		state->probPath.SetZeroAll();
		state->nstep = 0.0F;
		state->endMark.SetZeroAll();

		state->stateNum = 0;
	}

	/*
	read a state
	>> model - the  model that keeps the network created so far
	>> state - a set of states. It keeps
	1) hypotheses (states)
	2) probabilities of hypotheses
	3) parts of the network for expanding toward the next state
	*/
	void S2TPredictor::Read(S2TModel* model, StateBundle* state)
	{
		m = model;
		s = state;
	}

	XTensor S2TPredictor::Suppress(XTensor& input)
	{
		XTensor modify;
		InitTensor3D(&modify, input.GetDim(0), 1, 1, X_FLOAT, input.devID);
		modify = ScaleAndShift(modify, 0.0, -1e9);

		if (suppressSymbolNum <= 0)
			return input;

		for (int i = 0; i < suppressSymbolNum; i++) {
			_SetDataIndexed(&input, &modify, input.order - 1, suppressSymbols[i]);
		}

		return input;
	}

	void S2TPredictor::Predict(StateBundle* next, XTensor& aliveState, XTensor& encoding,
		XTensor& inputEnc, XTensor& paddingEnc, int batchSize, bool isStart,
		XTensor& reorderState, bool needReorder, int nstep)
	{
		int dims[MAX_TENSOR_DIM_NUM];

		/* word indices of positions up to next state */
		XTensor inputDec;

		/* the first token */
		XTensor first;
		InitTensor1D(&first, startSymbolNum, X_INT, inputEnc.devID);
		first.SetData(startSymbols, startSymbolNum);
		first = Unsqueeze(first, 0, batchSize);

		/* add a new word into the input sequence of the decoder side */
		if (isStart) {
			inputDec = Identity(first);
		}
		else {
			/* only pass one step to the decoder */
			inputDec = GetLastPrediction(s, inputEnc.devID);
		}

		/* keep alive states for the decoder */
		if (aliveState.dimSize[0] < batchSize) {
			/* alive inputs */
			inputDec = AutoGather(inputDec, aliveState);

			/* alive cache */
			for (int i = 0; i < m->decoder->nlayer; i++) {
				m->decoder->selfAttCache[i].KeepAlive(aliveState);
				m->decoder->enDeAttCache[i].KeepAlive(aliveState);
			}
		}

		if (needReorder) {
			for (int i = 0; i < m->decoder->nlayer; i++) {
				m->decoder->selfAttCache[i].Reorder(reorderState);
				m->decoder->enDeAttCache[i].Reorder(reorderState);
			}
		}

		/* prediction probabilities */
		XTensor& output = next->prob;
		XTensor decoding;

		for (int i = 0; i < inputDec.order - 1; i++)
			dims[i] = inputDec.dimSize[i];
		dims[inputDec.order - 1] = inputDec.dimSize[inputDec.order - 1];

		XTensor maskDec;

		/* decoder mask */
		maskDec = m->MakeS2TTriMaskDecInference(batchSize, inputDec.GetDim(-1));

		/* make the decoding network */
		if (m->config->model.decPreLN)
			if (isStart)
				decoding = m->decoder->RunFastPreNorm(inputDec, encoding, &maskDec, NULL, nstep);
			else
				decoding = m->decoder->RunFastPreNorm(inputDec, encoding, NULL, NULL, nstep);
		
		/* TODO
		else
			decoding = m->decoder->RunFastPostNorm(inputDec, encoding, &maskEncDec, nstep);*/

		CheckNTErrors(decoding.order >= 2, "The tensor must be of order 2 or larger!");

		/* generate the output probabilities */
		bool outputSoftmax = false;
		if (outputSoftmax)
			output = m->outputLayer->Make(decoding, false);
		else
			output = MMul(decoding, X_NOTRANS, *m->outputLayer->w, X_TRANS);

		/*only consider the last token*/
		output = SelectRange(output, 1, output.GetDim(1) - 1, output.GetDim(1));

		output = Suppress(output);
		
		output = LogSoftmax(output, -1);
	}

}