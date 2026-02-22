# ECE408 Final Project - GPT-2

## Milestone 1

## Table of Contents

- [Introduction](#introduction)
- [Milestone Requirements](#milestone-requirements)
- [Implementation and Testing](#implementation-and-testing)
- [Code Submission and Grading](#code-submission-and-grading)
- [Profiling-and-Performance-Analysis](#profiling-and-performance-analysis)
- [Deliverables](#deliverables)

## Introduction

This milestone focuses on implementing a basic version of the GPT-2 model's forward pass kernels on the GPU using CUDA. You will focus on writing a simple functionally correct GPU kernel for each step of the forward pass without worrying about optimizing your code for performance. You may also try to get comfortable with using NVIDIA's profiling tools to identify performance bottlenecks and areas for optimization, though this will not be a requirement for this milestone. 

## Milestone Requirements

For this milestone, you will need to write a basic parallel implementation for all forward pass kernels in the `kernels` folder. Before completing these kernels, you should make sure that you have a good conceptual understanding of the GPT-2 transformer architecture. For example, you should understand what each kernel does and the role that each plays in the forward pass of the GPT-2 model. We may ask you conceptual questions related to the GPT-2 architecture as part of this milestone's demo. (More information on the demo is in [README.md](README.md)).

**Note:** Your implementation needs to include the host code that launches your kernels, but you do NOT need to allocate/copy memory for the inputs/outputs. This means that you may assume memory management and copying is done properly by the provided code outside of the "`xxxx_forward()`" functions, and all you need to write in them is code that launches your kernel(s). You **should not** make changes to the function signatures of the provided "`xxxx_forward()`" **host functions**, but you may modify the **kernel function signatures** (or add additional kernels that your host functions calls) as needed. 

**Important:** 
 - We expect you to make consistent and frequent pushes to your repository. This will help us understand and track your progress.
 - You should also ensure that your code follows good coding conventions. For example, any unclear or unusual code/tricks should be explained with comments, and variable names should generally be descriptive or clear from context. Your grade will depend on your thorough understanding of the code and code clarity. 


## Implementation and Testing

### Testing Individual Kernels 

For this milestone, we provide a few different test scripts for you to verify the correctness of each of your forward pass kernels, as well as your overall inference output correctness. 


#### output_verification_rand
The file `output_verification_rand.cu` contains some skeleton test code that allows you to quickly test the basic functionality of each kernel you wrote by feeding in randomly generated inputs and comparing the outputs of your kernels to the provided CPU kernels. We encourage you to modify this for your own debugging purposes.

In this script, the tests for each kernel are run independently, which means the test result of each kernel only depends on the correctness of that kernel, and errors in one kernel does not affect the correctness of the other kernels. 

**NOTE:** This file is provided to quickly get you started on debugging your individual kernel implementations. You need to make some small changes to the script before you start (marked with "TODO: "). We recommend using a lenient error_threshold when you first start testing, since this test script only compares outputs from the provided reference **CPU** code to that of the **GPU** code. (Why one shouldn't naively set the error threshold to some super small value or 0 to ensure "correctness" is left to the reader as an exercise). You should **NOT** try to use this test script to guarantee the *full correctness* of your kernels.

To use this test script, run the command

    make output_verification_rand

then you can run the test script by uncommenting the line `srun ./output_verification_rand` in the `verify.slurm` file and then running.

    sbatch verify.slurm

The outputs of the testing script will be saved in the `verify_output.out` file. 

To clean, run 

    make clean

This will remove all the files generated during the compilation and execution process.

#### test_gpt2_kernels

The script `test_gpt2_kernels.cu` tests the functionality of your kernels by simulating a forward pass and comparing the outputs of your kernels at each step to the outputs of our gold forward pass kernel implementations (which runs on the GPU).

To use this test script, run the command

    make test_gpt2_kernels

then you can run the test script by uncommenting the line `srun ./test_gpt2_kernels` in the `verify.slurm` file and then running.

    sbatch verify.slurm

The outputs of the testing script will be saved in the `verify_output.out` file.

### Overall Inference Output Testing

After you are (somewhat) confident with all of your kernel implementations, you can run the provided full inference test script to verify the correctness of your GPT-2 forward pass implementation.

Since the full forward pass requires all of the implemented kernels to be correct, you should only run this test after you have verified the correctness of all of your kernels.

To build your implemented kernels and verify the output of the full forward pass of the GPT-2 model, run the command

    make test_gpt2

Then, make sure the line `srun ./test_gpt2` is uncommented in the `verify.slurm` file and run

    sbatch verify.slurm

The outputs of the testing script will be saved in the `verify_output.out` file.

### Talk to the GPT2 You Wrote!

Once you have implemented the forward pass correctly, you will be able to talk* to the GPT-2 model you wrote! You can input text into the model and have it complete the text for you. To do this, run the command

    make next_token_generation

Then, go into `generate_tokens.slurm` and edit the text you want the model to complete. After that, run

    sbatch generate_tokens.slurm
   
Once the job has completed, you can then find the model outputs in `generate_tokens_output.out`!

*\*talk: voice recognition not included, and no fancy chatbot capabilities, but by inputting text and having the model complete it, you can still technically talk to the GPT-2 model you wrote!*

*Also note that the tokenizer is not fully implemented. Prompts with simpler words/vocabulary should work fine, but more uncommon words/slangs or complex symbols in the input may result in non-ideal output.*

## Code Submission and Grading

To submit your code, push to the main branch of your group's repository. 
Only code in the `kernels` folder will be graded. 

The grade breakdown for the coding portion of this milestone is as follows:

| Kernel                        | Weight |
|-------------------------------|------- |
| Attention (including Softmax) | 5%     |
| Encoder                       | 1%     |
| LayerNorm                     | 2%     |
| MatMul                        | 2%     |
| Residual                      | 1%     |
| GELU                          | 1%     |

*Total: 12% of project grade*

## Profiling and Performance Analysis   

You may use NVIDIA's profiling tools to identify performance bottlenecks and areas for optimization. Although profiling is **not required** in this milestone, it is highly recommended to start profiling your baseline implementation once you're done implementing the kernels. You will be asked to profile your baseline implementation in Milestone 2.

Given that you have implemented a MVP (Minimum Viable Product) of the GPT2 forward pass, you should be able to profile the performance of your implementation and identify bottlenecks and potential areas for optimization.

**Before you do any profiling, make sure your implementation is functionally correct. Also make sure you do not have any memory errors by running *compute-sanitizer*.** 
This tool is already installed and available in the delta environment. To use it, simply add `compute-sanitizer` after `srun`. For more information, see the [NVIDIA compute-sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). 

For this course, we're using Delta's A40x4 nodes, where each node has four A40 GPUs. Since each job submitted to Delta usually uses only one GPU, a single node can handle up to four jobs simultaneously. However, when profiling your code, it's important to avoid external interference. **Therefore, you need to ensure that you're adding the `perf,nvperf` constraints in your slurm file to ensure exclusive access to the node.** The trade-off is longer wait times if the cluster is busy. It's recommended to remove `perf,nvperf` when testing the correctness of your code, and include it only during profiling.

The top of your slurm script should look like this when profiling:

```bash
#!/bin/bash
#SBATCH --job-name="verify"
...
#SBATCH --constraint="projects,perf,nvperf"
...
```


***System level profiling using Nsight-Systems***

`nsys` (Nsight Systems) profiles the execution at the system/application level. You can generate a profile using the `nsys` command provided in the `verify.slurm` file.

You can also download the Nsight Systems application, where you can visualize the timeline of your application's execution. Download the `.nsys-rep` file generated from delta onto your local system and open it in the Nsight Systems app to visualize the timeline.

You can find more information about nsys in the [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling).

***Kernel level profiling using Nsight-Compute***

Please not that Nsight-Systems does not give you detailed kernel level performance metrics. For that, we will need to use `ncu` (Nsight-Compute).

You will need to install NVIDIA NSight Compute on your own machine. It can be downloaded from NVIDIA's [website](https://developer.nvidia.com/nsight-compute) as a standalone application.

After installing NSight Compute, you can profile your code by running the command provided in the slurm file.

You can open the generated `.ncu-rep` file in the Nsight Compute application. Nsight Compute provides detailed information about the performance of each kernel, including compute latency/throughput metrics, memory workload analysis, and more. 

## Deliverables


| Step | Deliverables                                             |
| ---- | ------------------------------------------               |
| 1    | Implemented forward pass kernels in the `kernels` folder |
| 2    | Knowledge of the GPT-2 architecture                      |
| 3    | M1 Demo Meeting                                          |


We are excited to hear from you soon!
