/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
* All rights reserved.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-01
 * $Updated by: Yuhao Zhang (email: yoohao.zhang@gmail.com) 2023-09-20
 */

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "Conv1D.h"
#include "cudnn.h"


namespace nts { // namespace nts(NiuTrans.Tensor)

/*
* 
*/
void _Conv1DBase(const XTensor *input, const XTensor *weight, const XTensor *bias, XTensor *output,
                      int stride, int padding, bool useBias)
{
    CheckNTErrors((input->GetDim(0) == output->GetDim(0) &&
                   input->GetDim(1) == weight->GetDim(1) &&
                   output->GetDim(1) == weight->GetDim(0)),
                  "Unmatched dimension");

    CheckDev(input->devID, weight->devID);
    if (useBias) {
        CheckNTErrors(bias->unitNum == weight->GetDim(0), "Unmatched dimension of bias")
        CheckDev(weight->devID, bias->devID);
    }
    int blockNum = input->GetDim(0);
    int inChannel = input->GetDim(1);
    int outChannel = output->GetDim(1);
    int inputLength = input->GetDim(2);
    int kernelSize = weight->GetDim(2);
    int blockStride = output->GetDim(2);
    int inputBlockSize = input->GetDim(1) * input->GetDim(2);
    int outputBlockSize = output->GetDim(1) * output->GetDim(2);


    DTYPE* ip = (DTYPE*)input->data;
    DTYPE* op = (DTYPE*)output->data;
    DTYPE* wp = (DTYPE*)weight->data;
    DTYPE* bp = NULL;
    if (useBias)
        bp = (DTYPE*)bias->data;
#ifdef USE_CUDA
    #ifdef USE_CUDNN
    if (input->devID >= 0 || weight->devID >= 0 || (bias != NULL && bias->devID >= 0)) {

        cudaSetDevice(input->devID);
        //Declare cudnn handle 
        cudnnHandle_t cudnnHandle;
        CheckCUDNN(cudnnCreate(&cudnnHandle));

        //Declare input shape
        cudnnTensorDescriptor_t inputDesc;
        CheckCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
        CheckCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              input->dimSize[0],  // batch
                                              input->dimSize[1],  // input chanel
                                              1,                  // placeholder
                                              input->dimSize[2]));// length
        //Declare cnn kernel shape
        cudnnFilterDescriptor_t kernelDesc;
        cudnnConvolutionDescriptor_t convDesc;
        CheckCUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
        CheckCUDNN(cudnnSetFilter4dDescriptor(kernelDesc,
                                              CUDNN_DATA_FLOAT,
                                              CUDNN_TENSOR_NCHW,
                                              weight->dimSize[0],  // input chanel
                                              weight->dimSize[1],  // output chanel
                                              1,                   // placeholder
                                              weight->dimSize[2]));// kernel


        //Declare cnn operation
        CheckCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        CheckCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, padding, // padding
                                                   1, stride, // stride
                                                   1, 1,  //dilation
                                                   CUDNN_CROSS_CORRELATION, //CUDNN_CONVOLUTION
                                                   CUDNN_DATA_FLOAT));
        CheckCUDNN(cudnnSetConvolutionGroupCount(convDesc, 1)); //group

        //Declare output shape
        cudnnTensorDescriptor_t outputDesc;
        CheckCUDNN(cudnnCreateTensorDescriptor(&outputDesc));
        CheckCUDNN(cudnnSetTensor4dDescriptor(outputDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              output->dimSize[0],
                                              output->dimSize[1],
                                              1,
                                              output->dimSize[2]));

        //Get the output shape to check
        int checkBatchSize = 0, checkOutputChannels = 0, placeholder = 0, checkOutputLenth = 0;
        CheckCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                         inputDesc,
                                                         kernelDesc,
                                                         &checkBatchSize,
                                                         &checkOutputChannels,
                                                         &placeholder,
                                                         &checkOutputLenth));

        //Check parameters
        CheckNTErrors( (output->dimSize[2] == checkOutputLenth) &&
            (output->dimSize[1] == checkOutputChannels), "The output shape is not correct!");

        CheckNTErrors(cudnnHandle != nullptr && inputDesc != nullptr && kernelDesc != nullptr &&
            convDesc != nullptr && outputDesc != nullptr, "Some cudnn settings are missing!");


        //Declare CNN forward algorithm
        int cudnnBestCNNAlgo=0;
        int numAlgos = 1;
        std::unique_ptr<cudnnConvolutionFwdAlgoPerf_t[]> perf_algos(
            new cudnnConvolutionFwdAlgoPerf_t[numAlgos]);
        int returned_algo_count{ 0 };
        CheckCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
                                                          inputDesc,
                                                          kernelDesc,
                                                          convDesc,
                                                          outputDesc,
                                                          numAlgos,
                                                          &returned_algo_count,
                                                          perf_algos.get()));
        //Allocate the workspace memory
        cudnnBestCNNAlgo = perf_algos[0].algo;
        size_t workspaceSize = 0;
        CheckCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                           inputDesc,
                                                           kernelDesc,
                                                           convDesc,
                                                           outputDesc,
                                                           cudnnConvolutionFwdAlgo_t(cudnnBestCNNAlgo),
                                                           &workspaceSize));
        void* workspace = nullptr;
        cudaMalloc(&workspace, workspaceSize);

        //Start to calculate
        auto alpha = 1.0f, beta = 0.0f;
        CheckCUDNN(cudnnConvolutionForward(cudnnHandle,
                                           &alpha,
                                           inputDesc, input->data,
                                           kernelDesc, weight->data,
                                           convDesc, cudnnConvolutionFwdAlgo_t(cudnnBestCNNAlgo),
                                           workspace, workspaceSize,
                                           &beta,
                                           outputDesc, output->data));

        //Compute the bias if necessary
        if (useBias) {
            int dimOp = 1;
            _SumDim(output, bias, output, dimOp, (DTYPE)1.0);
        }

        return;
    }
    else {
        // TODO!! Do not use cudnn.
        ShowNTErrors("TODO!");
    }
    #endif
    ShowNTErrors("TODO!");
#endif


    DTYPE value;
    for (int i = 0; i < blockNum; i++) {
        DTYPE * tmpIp = ip + i * inputBlockSize;
        DTYPE * tmpOp = op + i * outputBlockSize;

//        for (int j = 0, index=-padding; j < blockStride; j++, index += stride) {
//            for (int k = 0; k < outChannel; k++) {
//                DTYPE * tmpWp = wp + k * inChannel * kernelSize;
//
//                value = 0;
//                int tmpIndex = index;
//                for (int x = 0; x < kernelSize; x++) {
//                    for (int y = 0; y < inChannel; y++) {
//                        if (tmpIndex >= 0 && tmpIndex < inputLength)
//                            value += tmpIp[y * inputLength + tmpIndex] * tmpWp[y * kernelSize + x];
//                    }
//                tmpIndex++;
//                }
//
//                if (bp != NULL)
//                    value += bp[k];
//                tmpOp[k * blockStride + j] = value;
//            }
//        }

        for (int j = 0; j < blockStride; j++) {
            for (int k = 0; k < outChannel; k++) {
                DTYPE * tmpWp = wp + k * inChannel * kernelSize;

                value = 0;
                int index = j * stride - padding;

                for (int x = 0; x < kernelSize; x++) {
                    for (int y = 0; y < inChannel; y++) {
                        if (index >= 0 && index < inputLength)
                            value += tmpIp[y * inputLength + index] * tmpWp[y * kernelSize + x];
                    }
                    index++;
                }

                if (bp != NULL)
                    value += bp[k];
                tmpOp[k * blockStride + j] = value;

            }
        }
    }
}

/*
TODO
*/
XTensor Conv1DBias(const XTensor &input, const XTensor &weight, const XTensor &bias,
                   int stride, int padding, bool useBias)
{


    CheckNTErrors((input.order==3 && weight.order==3), "The orders of input and weight should be 3");
    
    XTensor output;

    int outputLength = ((input.GetDim(2) + 2 * padding - weight.GetDim(2)) / stride) + 1;

    InitTensor3D(&output, input.GetDim(0), weight.GetDim(0), outputLength, X_FLOAT, input.devID);
    output.SetTMPFlag();

    _Conv1DBase(&input, &weight, &bias, &output, stride, padding, useBias);

    /* tensor connections */
    TensorList params(2);
    params.Add((XTensor*)&weight);
    if (useBias)
        params.Add((XTensor*)&bias);

    if (input.enableGrad && output.enableGrad) {
        XLink::MakeLink(&params, &output, MATH_CONV1D);
        XLink::AddParamToHeadInt(&output, stride);
        XLink::AddParamToHeadInt(&output, padding);
    }


    return output;

}

XTensor Conv1DBase(const XTensor& input, const XTensor& weight, int stride, int padding)
{


    CheckNTErrors((input.order == 3 && weight.order == 3), "The orders of input and weight should be 3");

    XTensor output;

    int outputLength = (input.GetDim(2) + 2 * padding - weight.GetDim(2) / stride) + 1;

    InitTensor3D(&output, input.GetDim(0), weight.GetDim(0), outputLength, X_FLOAT, input.devID);
    output.SetTMPFlag();

    _Conv1DBase(&input, &weight, NULL, &output, stride, padding, false);

    /* tensor connections */
    TensorList params(2);
    params.Add((XTensor*)&weight);

    if (input.enableGrad && output.enableGrad) {
        XLink::MakeLink(&params, &output, MATH_CONV1D);
        XLink::AddParamToHeadInt(&output, stride);
        XLink::AddParamToHeadInt(&output, padding);
    }


    return output;

}

} // namespace nts(NiuTrans.Tensor)
