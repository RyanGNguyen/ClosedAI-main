# ECE408 Final Project - GPT-2

## Milestone 3

## Table of Contents

* [Introduction](#introduction)
* [Milestone Requirements](#milestone-requirements)
* [Implementation and Testing](#implementation-and-testing)
* [Performance Analysis and Profiling ](#performance-analysis-and-profiling)
* [Final Report](#final-report)
* [Deliverables](#deliverables)
* [Submission and Grading](#submission-and-grading)
* [Additional Details/Hints for Required Optimizations](#additional-detailshints-for-required-optimizations)
    + [req_4: Flash Attention](#req_4-flash-attention)
    + [req_5: Configuration Sweep/Optimization](#req_5-configuration-sweepoptimization)
    + [req_6: Constant Memory](#req_6-constant-memory)
    + [req_7: `__restrict__`](#req_7-restrict)
    + [req_8: Local/Windowed Attention](#req_8-localwindowed-attention)
    + [req_9: Split-K](#req_9-split-k)

## Introduction

This milestone focuses on further optimizing the GPU kernels from Milestone 2. Your aim is to create the world's best possible implementation of the GPT-2 model's forward pass kernels on the GPU. 
This is where you let your imaginations fly and CUDA/ML knowledge shine! 
You will also profile your optimizations and present your findings in a final report and presentation, with the option of writing a detailed and focused tech blog to replace either the report or presentation.

## Milestone Requirements

- Complete the below list of required optimizations
- Implement proposed optimizations from Milestone 2
- Profiling:
  - Find each individual required and proposed optimization's effectiveness through various profiling metrics
  - Determine **specific reasons** for speedups/slowdowns/no effect
  - Describe how different optimizations work together (or against each other)
  - Compare the final optimized implementation with baseline kernels from Milestone 1
- Present findings:
  - Final Report

### Required optimizations:

| Optimization | Description                                  |
| ------------ | -----------                                  |
| req_4        | Flash Attention                              |
| req_5        | Configuration Sweep/Optimization             |
| req_6        | Constant Memory                              |
| req_7        | `__restrict__`                               |
| req_8        | Local/Windowed Attention                     |
| req_9        | Split-K                                      |

**Important**: Please see additional details/requirements for some of the required optimizations provided at the end of this document.

### Proposed optimizations:

In addition to the required optimizations, you are also required to implement **all** proposed optimizations from your Milestone 2 proposal. It's ok if you end up with implementation details that differ from your proposed optimizations, but you should make sure to explain and justify any deviations in your final report. These proposed optimizations will be graded manually during and after your Milestone 3 demo, and you should be prepared to explain your implementation choices and performance analysis during the demo. 

**Note**: You may also implement additional optimizations beyond those in your Milestone 2 proposal if you wish. Additionally, you may choose to have any new optimizations (ones you did **not** propose in Milestone 2) graded as part of your proposed optimizations. *You may want to do this if you decide you are not able to implement some of the optimizations you proposed, but don't want to lose all proposed optimization correctness points.* However, if you choose to do so, you will need to explain and justify these additional optimizations in your final report **in addition to all of your original proposed optimizations** to receive full credit on your final report.

## Implementation and Testing

For this milestone, all required optimizations **except for** req_8 (Local/Windowed Attention) will be graded on accuracy/correctness using the existing verification scripts (`test_gpt2_kernels.cu` or `test_gpt2.cu`). However, since you will also implement your proposed optimizations, your group is ultimately responsible for demonstrating the correctness of any optimizations that causes deviations from our provided op-level and end-to-end solution values. In other words, if a proposed optimization you implement in this milestone fails tests in either `test_gpt2_kernels.cu` or `test_gpt2.cu`, you will need to develop a sufficient verification method (for example, using the *Perplexity* scoring system to evaluate the model's language modeling accuracy) and justify in your final report why the optimization is implemented correctly, otherwise you **may NOT** receive full credit. For proposed optimizations that do not cause deviations from our provided op-level and end-to-end solution values, the existing verification scripts (`test_gpt2_kernels.cu` and `test_gpt2.cu`) are sufficient. 

Since req_8 (Local/Windowed Attention) changes the attention mechanism from full attention to local/windowed attention, it is expected that this optimization will cause deviations from our provided solution values in `test_gpt2_kernels.cu` and `test_gpt2.cu`. See the additional details section for more information on how to verify this optimization.

## Performance Analysis and Profiling 

Refer to [M2_README.md](M2_README.md) for profiling information. 

## Final Report

After performing both system-level profiling and kernel-level profiling, your team will need to submit a final report detailing the following:
- **Implementation Details**
    - For each optimization, describe how you implemented it.
    - For each proposed optimization, also describe the motivation behind them.
    - If your proposed optimizations did not yield performance improvements, explain why you think that is the case.
    - Explain how you verified each implementation's correctness (especially for proposed optimizations, if the final model outputs were different).
    - Any challenges you faced during implementation and how you overcame them.

- **Performance Analysis of Milestone 3 Optimizations**
    - Compare the performance of your final optimized kernels with the baseline implementations from Milestone 1.
    - Be sure to cover **all** new Milestone 3 optimizations (required and proposed).
    - Analyze why each optimization was effective (or why it wasn't).
    - What bottlenecks are addressed by your final version of each kernel?
    - Logical conclusions should be drawn in addition to just reporting metrics.
    - Detailed metrics and graphs can be used to support your analysis. 
    - End-to-end performance comparisons should also be included here.

- **Pretty Pictures and Data**
    - We'd really appreciate some nice charts or graphs of performance metrics! (annotate them and use them to supplement your analysis instead of a raw screenshot would be even better)
    - Comparison tables for things like execution time can be helpful too. 

- **Interesting Findings**
    - Anything you find interesting and/or surprising from the profiling results.

When writing this report, you should make full use of the detailed metrics provided by Nsight-Systems and Nsight-Compute. 

You should be detailed and in-depth about your findings and comparisons from profiling, **do NOT** just copy some metrics from Nsight, explain what those metrics are, and call it a day! Make sure to explain and provide insights on **why** the metrics look the way they do, and what implementations decisions led to those metrics.

This report does not have a length requirement. A long report packed with different metrics and surface-level analysis is **not** what we're looking for (*that's the job of the Nsight tools*). Your job is to interpret the data and provide **concise yet insightful conclusions**. This means we prefer quality over quantity, and some good insights on only a couple kernels is better than a bunch of surface-level analysis/boring metrics on every kernel.

Your report should be a document that records all high level design decisions, optimizations, and performance analysis done throughout the entire GPT Final Project. The report should be written in a way that is accessible to a general audience, but also contains details that would allow you (or anyone else) to reproduce your results if you look back to this project 5 years later. 

## Deliverables

| Step | Deliverables                                     |
| ---- | ------------------------------------------       |
| 1    | Implemented M3 Required Optimizations            |
| 2    | Implemented Proposed Optimizations               |
| 3    | Profiling and Performance Analysis               |
| 4    | Milestone 3 Demo                                 |
| 5    | Final Report                                     |

## Submission and Grading

To submit your final report, push it as a **PDF file** to the root directory of your GitHub main branch named `[team_name]_Final_Report.pdf`, where `[team_name]` is your team's name. There is no strict format requirement for them as it will be manually graded, but make sure the file is typed (not hand-written) and correctly named.

### Code Submission

For each of the required optimizations, you should have a folder named `kernels_req_x`, where x is the optimization number (e.g. `kernels_req_4` for optimization 4). They should be submitted in a similar format to Milestone 2 code submission in order to interface with the autograder correctly. Refer to Milestone 2 submission format for details. The exception is `kernels_req_5`, where you should put the configuration sweep script(s) that you used. 

Your proposed optimizations will be graded manually during and after the milestone 3 demo. You should submit proposed optimizations in separate git branches from the main branch in your repository (you may decide how you want to organize your proposed optimizations, but the submission format for the required optimizations in the main branch should be unaffected.)

For example, the submission for required optimization 4 should be in a folder called `kernels_req_4` and contains its own copy of **all** code/scripts needed to run and verify that optimization (any files in the `kernels/` folder, additional helper files you make, etc):
```
 fa25_ece408_[team_name]
  ├── kernels
  │   ├── attention.cuh
  │   ├── encoder.cuh
  │   ├── gelu.cuh
  │   ├── matmul.cuh
  │   └── ...          [unmodified kernels from M1]
  ├── kernels_req_4
  │   └── attention.cuh 
  ├── kernels_req_5
  │   └── [configuration sweep script]
  ├── kernels_req_6
  │   └── ...
  ├── kernels_req_7
  │   └── ...
  ├── kernels_req_8
  │   └── ...
  ├── kernels_req_9
  │   └── ...
  ├── gpt2.cuh
  ├── Makefile
  ├── README.md
  ├── [team_name]_M2_Profiling_Report.pdf
  ├── [team_name]_Final_Report.pdf
  └── ...
```

The grade breakdown for the coding portion of this milestone is as follows:

| Required Optimization                 | Weight |
|-------------------------------        |------- |
| req_4: Flash Attention                | 6%     |
| req_5: Configuration Sweep            | 3%     |
| req_6: Constant Memory                | 2%     |
| req_7: `__restrict__`                 | 1%     |
| req_8: Local/Windowed Attention       | 2%     |
| req_9: Split-K                        | 3%     |


*Total: 17% of project grade*

For **proposed optimizations**, which is 10% of the project grade, correctness points (and potentially extra credit) awarded per optimization will be based on the difficulty of each proposed optimization. During your Milestone 3 demo, you will be asked to confirm which proposed optimizations you have implemented for correctness points. 

**Important**: Any incorrect/non-functioning implementations will receive 0 correctness points. However, you may still earn points for your understanding and analysis of the optimization in your final report even if the implementation is incorrect/non-functioning.


## Additional Details/Hints for Required Optimizations


### req_4: Flash Attention

The FlashAttention series of papers proposed a monumental IO-aware attention optimization that is now widely used in the industry. The algorithm enables tiling of the attention matrix to address the memory bandwidth bottleneck in attention. You may choose to implement either the basic FlashAttention algorithm, outlined in the original [FlashAttention paper](https://arxiv.org/pdf/2205.14135), **or** the improved version, outlined in the [FlashAttention2 paper](https://tridao.me/publications/flash2/flash2.pdf). 

**Note:** Both FlashAttention and FlashAttention2 are long papers that may take a while to read fully, so we recommend focusing on the following sections:
  - If you choose to implement the basic FlashAttention: Introduction, Section 2.2 (Standard Attention Implementation), Section 3.1 (**Skip** discussions of Recomputation)
  - If you choose to implement FlashAttention2: Introduction, Section 2.2 (Standard Attention Implementation), Section 2.3.1 (Basic FlashAttention Algorithm), Section 3.1.1 (FlashAttention2 Algorithm). 
  
    *If you are using the FlashAttention2 paper on arxiv.org, there may be a minor mistake in the pseudo code outlined in section 3.1.1. Please use the paper link provided above to ensure you have the correct version.*

The papers have already provided pseudo code for the algorithms, your job is to translate them into efficient CUDA code.

### req_5: Configuration Sweep/Optimization

For this optimization, create a script (python, bash, or any language of your choice) to systematically sweep through different configurations like block sizes, thread counts, loop unrolling, etc. 
*Note for loop unrolling: Loop unrolling can technically be applied to any loop in your code, but you should understand where you should focus on and why. Make sure to perform thorough profiling after implementing this optimization and be able to directly point out the effects of loop unrolling.*

Make sure you at least provide why and how you are sweeping each configuration, what you expected before the sweep, and what you found after the sweep. You should try a sufficient amount of configurations to ensure you are getting the best performance from this optimization. 

In short, you should be able to answer the following questions:
- How does your script sweep through different configurations?
- What configurations did you try?
- Why did you choose to focus on these specific parameters?
- What are your findings? Include expectations before and analysis/conclusion after.

**Make sure to submit your sweeping script by pushing it to the main branch of your group's repository.** Note that we will not grade this script. However, we will manually check that you have submitted a working sweeping script in the `kernels_req_5` folder. You should also refer this script in your final report's when discussion this optimization.

Note that there is not a minimum number of configurations you need to try, but make sure you try enough configurations to ensure you are getting the best performance from your kernels, and sufficient justifications of your choices and analysis in your final report will earn you full points for this optimization.

### req_6: Constant Memory

Think about what data you can/should store in constant memory. Review the lecture covering constant memory if needed. Make sure to perform thorough profiling after implementing this optimization to see how it is actually improving performance (*or explain why it's not*). 

You may choose which kernel(s) to apply this optimization to, and providing sufficient justifications of your choices and analysis in your final report will earn you full points for this optimization.

### req_7: `__restrict__`

Make sure you understand what `__restrict__` does and how it can help improve performance before starting. You should also be able to explain why (and where) you are using `__restrict__` in your code.

You may choose which kernel(s) to apply this optimization to, and providing sufficient justifications of your choices and analysis in your final report will earn you full points for this optimization.

### req_8: Local/Windowed Attention

We have provided a reference CPU implementation in `cpu_kernels/local_attention.cuh` for this optimization. Make sure you understand what exactly local/windowed attention is based on the provided CPU code before starting this optimization. 
Additionally, we have provided a basic skeleton test script (*which you need to modify yourself*) for this optimization (`local_attn_verify.cu` and `local_attn_verify.slurm`). You may refer to these files when implementing and verifying your local/windowed attention optimization. 
**Important:** For grading and verification purposes, you need to implement submit using a **window size of 128** (as shown in `local_attention.cuh`). However, once you have verified your implementation with a window size of 128 using our provided test data, feel free to experiment with different window sizes and include your findings in the final report.
**Note:** Make sure to use `make local_attn_verify` to compile your code base instead of simply using `make` to ensure your modifications to the local attention verification script is compiled correctly.

### req_9: Split-K

Split-K is a technique that splits the K dimension of a GEMM operation into smaller chunks, allowing for better parallelism and memory access patterns. This can lead to improved performance, especially for certain matrix sizes. For more details on Split-K, you may refer to the advanced optimizations lecture (Lecture 16). 

You may choose which kernel(s) to apply this optimization to. Correct implementation of this optimization to **at least 1 kernel** will earn you full points for this optimization, but you **need** to perform thorough profiling and provide detailed analysis in your final report on how this optimization is affecting performance and justify the implementation choices you made.