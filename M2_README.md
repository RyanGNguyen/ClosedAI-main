# ECE408 Final Project - GPT-2

## Milestone 2

## Table of Contents

- [Introduction](#introduction)
- [Milestone Requirements](#milestone-requirements)
- [Implementation and Testing](#implementation-and-testing)
- [Performance Analysis/Profiling](#performance-analysisprofiling)
- [Profiling Report and Optimization Proposal](#profiling-report-and-optimization-proposal)
- [Submission and Grading](#submission-and-grading)
- [Deliverables](#deliverables)
- [Additional Details/Hints for Optimizations](#additional-detailshints-for-optimizations)

## Introduction

This milestone focuses on profiling and optimizing some of the GPU kernels from Milestone 1, and using your profiling results to guide you to propose some of your own optimizations for Milestone 3.

## Milestone Requirements

For this milestone, you will need to: 
- Complete the below list of required optimizations.
- Profile your M1 baseline kernel implementations as well as all M2 required kernel optimizations.
- Present your profiling findings in the form of a report.
- Analyze your results to motivate additional optimizations you plan to implement in the next milestone, and propose them in your report.

### Milestone 2 Required Optimizations:
| Optimization | Description                                                        |
| ------------ | -----------                                                        |
| req_0        | Joint Shared Memory and Register Tiling for Matrix Multiplication  |
| req_1        | Tensor Core for Matrix Multiplication                              |
| req_2        | Utilize cuBLAS                                                     |
| req_3        | Reduction                                                          |

**Note:** You may **NOT** use cuBLAS in req_1 (Tensor Core) or req_0 (Register Tiling). 

## Implementation and Testing

Refer to M1_README.md for instructions on how to use the provided testing scripts; you may use the same ones from Milestone 1. Note that for `req_1` and `req_2` you will do some of your calculations using lower precision, so your errors may be higher than in Milestone 1. 

## Performance Analysis/Profiling 

In this milestone, You are **required** to use NVIDIA's profiling tools to identify performance bottlenecks and areas for optimization. 

Given that you have implemented a MVP (Minimum Viable Product) of the GPT-2 forward pass, and several additional optimizations, you should be able to profile the performance of your implementation and identify bottlenecks in your baseline implementation and draw comparisons with your implemented optimizations. 

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

`nsys` (Nsight Systems) profiles the execution at the system/application level. You can generate a profile using the `nsys` command provided in the slurm file.

After running the command, you should see something that looks like the following (but not identical) in the generated `..._nsys_prof.out` file:

```bash 
......

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)          Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  -----------  ---------------------
     99.9  351,122,724,860      3,519  99,779,120.4  100,089,303.0     2,855  100,130,281  5,413,528.2  poll                 
      0.1      283,382,530        925     306,359.5       14,207.0     1,051   20,208,549  1,050,067.9  ioctl                
     ......               
      0.0            1,913          1       1,913.0        1,913.0     1,913        1,913          0.0  bind                 

[5/8] Executing 'cudaapisum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     ......     

[6/8] Executing 'gpukernsum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      GridXYZ         BlockXYZ                                               Name                                          
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ---------------  --------------  ----------------------------------------------------------------------------------------
     ......                                                                   

[7/8] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)       Operation     
 --------  ---------------  -----  -------------  -------------  -----------  -----------  ------------  ------------------
     ......

[8/8] Executing 'gpumemsizesum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)   StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  ---------  -----------  ------------------
     ......

```
The CUDA API Statistics section shows the CUDA API calls that are executed. The CUDA Kernel Statistics lists all the kernels that were executed during the profiling session. There are also more details on the CUDA memory operations (CudaMemcpy) listed. There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls.

In the Nsight Systems application, you can visualize the timeline of your application's execution. Download the `.nsys-rep` file generated onto your local system and open it in the Nsight Systems app to visualize the timeline.

You can find more information about nsys in the [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling).

***Kernel level profiling using Nsight-Compute***

Nsight-Systems does not give you detailed kernel level performance metrics. For that, we will need to use `ncu` (Nsight-Compute).

You will need to install NVIDIA Nsight Compute on your own machine. It can be downloaded from NVIDIA's [website](https://developer.nvidia.com/nsight-compute) as a standalone application.

After installing Nsight Compute, you can profile your code by running the command provided in the slurm file.

You can open the generated `.ncu-rep` file in the Nsight Compute application. Nsight Compute provides detailed information about the performance of each kernel, including compute latency/throughput metrics, memory workload analysis, and more. 

**Note**: Although Nsight Compute provides some useful analysis information based on the profiling data and may make some recommendations for optimization, it is important for you to fully understand exactly what causes the performance inefficiencies and why the suggestions Nsight Compute gave you are significant (*or why not*).

## Profiling Report and Optimization Proposal

After performing both system-level profiling using Nsight-Systems and kernel-level profiling using Nsight-Compute, your team will need to submit a report detailing your profiling results. 
There is no strict structure requirement for this report, but your report should touch on at least the following topics:

- **Analysis of Baseline Implementation**
    - What are the most time-consuming kernels in your baseline implementation?
    - How's your baseline kernels' memory access pattern?
    - Some relevant metrics can be helpful.

- **Performance Improvements from Required Optimizations**
    - What's the performance difference between baseline and each optimization you implemented in this Milestone?
    - Analyze why each optimization was effective (or why it wasn't)
    - Detailed metrics can be used to support your analysis. 

- **Pretty Pictures and Data**
    - We'd really appreciate some nice charts or graphs of performance metrics! (annotate them and use them to supplement your analysis instead of a raw screenshot would be even better)
    - Comparison tables for things like execution time can be helpful too. 

- **Interesting Findings**
    - Anything you find interesting and/or surprising from the profiling results.

When writing this report, you should make full use of the detailed metrics provided by Nsight-Systems and Nsight-Compute. 

You should be detailed and in-depth about your findings and comparisons from profiling, **do NOT** just copy some metrics from Nsight, explain what those metrics are, and call it a day!

This report does not have a length requirement. A long report packed with different metrics and surface-level analysis is **not** what we're looking for (*that's the job of the Nsight tools*). Your job is to interpret the data and provide **concise yet insightful conclusions**. This means we prefer quality over quantity, and some good insights on only a couple kernels is better than a bunch of surface-level analysis/boring metrics on every kernel.

### Optimization Proposal
The logical next step after profiling your code thinking critically and coming up with you own optimizations. Following your profiling sections in your report, you will need to write up an optimization proposal section detailing how you plan on further optimizing the existing GPU forward pass kernels.
 
Your Proposal should contain around 2-4 additional optimizations you plan to implement in the next milestone. You may come up with your own ideas, analyze your profiling results, extend/expand an existing optimization (in a meaningful/nontrivial way), apply concepts from lecture, and/or search online for optimization ideas. Here are some ideas/options (*that aren't necessarily the easiest/best*): K-V Cache, Multi-GPU Support, NVIDIA CUTLASS, new algorithm to optimize a specific kernel, etc.

Note that these optimizations may **not** already be one of the M3 required optimizations that you will also implement in the next milestone. The M3 required optimizations are listed below for your reference.

You should provide a detailed description of the proposed optimizations, how you plan to implement them, and why you think they will be beneficial. Grading of this proposal will be based on effort, detail, and feasibility of the proposed optimizations. Keep in mind that you will need to implement these optimizations in the next milestone and they will be graded. 

#### Milestone 3 Required Optimizations (NOT for this milestone):

| Optimization | Description                                  |
| ------------ | -----------                                  |
| req_4        | Flash Attention                              |
| req_5        | Configuration Sweep/Optimization             |
| req_6        | Constant Memory                              |
| req_7        | `__restrict__`                               |
| req_8        | Local/Windowed Attention                     |
| req_9        | Split-K                                      |


## Submission and Grading

### Non-Code Submission
To submit your profiling report and optimization proposal, push them as **1 combined PDF file** to your GitHub main branch named `[team_name]_M2_Profiling_Report.pdf`, where `[team_name]` is your team's name. There is no strict format requirement for them as it will be manually graded, but make sure the file is typed (not hand-written) and correctly named.

### Code Submission

Your code should be submitted in the following manner:
  ```
 fa25_ece408_[team_name]
  ├── kernels
  │   ├── attention.cuh
  │   ├── encoder.cuh
  │   ├── gelu.cuh
  │   ├── matmul.cuh
  │   └── ...          [all unmodified kernels from M1]
  ├── kernels_req_0
  │   ├── matmul.cuh   [with the req_0 optimization, register tiling] 
  │   └── ...          [any other files to replace in kernels/ for req_0]
  ├── kernels_req_1
  │   ├── matmul.cuh   [with the req_1 optimization, tensor core]
  │   └── ...          [any other files to replace in kernels/ for req_1]
  ├── kernels_req_2
  │   ├── ...          [any files to replace in kernels/ for req_2]
  │   └── ...          [any files to replace in kernels/ for req_2]
  ├── kernels_req_3
  │   ├── ...          [any files to replace in kernels/ for req_3]
  │   └── ...          [any files to replace in kernels/ for req_3]
  ├── gpt2.cuh
  ├── Makefile
  ├── README.md
  ├── [team_name]_M2_Profiling_Report.pdf
  └── ...
  ```
You will submit each of your optimized kernels in separate folders named `kernels_req_x` where `x` is `0`, `1`, `2`, or `3`. During grading, when grading optimization `req_x`, we will copy any files from the `kernels_req_x` folder into the `kernels` folder, replacing any files with the same name. E.g., for `req_0`, `kernels_req_0/matmul.cuh` with the register tiling optimization will replace `kernels/matmul.cuh`, and those kernels will be run. 

For each optimization, you are expected to apply it everywhere it is applicable, within reason. If you are missing an expected file for an optimization, or it is not implemented as expected (manually checked), you will lose points on that optimization. It is your job as a team to figure out which kernels each optimization applies to, and include those files in the corresponding folder. You may however apply optimizations to additional kernels we did not expect; in fact we encourage you to do so. 

The grade breakdown for the coding portion of this milestone is as follows:

| Optimization                   | Weight |
|------------------------------- |------- |
| req_0: Register Tiling         | 6%     |
| req_1: Tensor Core             | 5%     |
| req_2: Utilize cuBLAS          | 4%     |
| req_3: Reduction               | 3%     |

*Total: 18% of project grade*

Please note that you will only receive full credit for each optimization if it is implemented correctly. Just like the lab assignments, there will **not** be partial credit for incorrectly implemented/non-functioning code.


## Deliverables

| Step | Deliverables                                     |
| ---- | ------------------------------------------       |
| 1    | Implemented M2 Required optimizations            |
| 2    | Profiling Results and Mini Report                |
| 3    | Optimization Proposal                            |
| 4    | M2 Demo Meeting                                  |

## Additional Details/Hints for Optimizations

### req_0: Joint Register and Shared Memory Tiling for Matrix Multiplication

There will be a lecture on advanced optimizations (Lecture 16), which will cover an overview of joint register and shared memory register tiling. Additionally, you can refer to [this](https://lumetta.web.engr.illinois.edu/508/slides/lecture4.pdf) ECE 508 lecture slide on the concept. **You are required to implement this optimization in all kernels where its use is reasonable.** As a hint, there is more than 1 kernel where register tiling can be reasonably used.


### req_1: Tensor Core for Matrix Multiplication
There will be a lecture on advanced optimizations (Lecture 16), which will cover an overview of using Tensor Cores for matrix multiplication. Note that you are **required** to implement tensor core with *TF32* precision. You may refer to the advanced optimizations lecture, as well as online NVIDIA documentation/blog posts for more information. **You are required to implement tensor core in all kernels where its use is reasonable.** As a hint, there is more than 1 kernel where tensor core can be reasonably used.


### req_2: Utilize cuBLAS

Make sure you understand how to properly setup and use cuBLAS. Throwing in a single function call to cuBLAS without understanding how it works may not earn you full points (and probably won't work). The [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/#using-the-cublas-api) may be helpful. **You are required to implement cuBLAS in all kernels where its use is reasonable.** As a hint, there is more than 1 kernel where cuBLAS can be reasonably used.


### req_3: Reduction

Here reduction refers to the algorithm covered in lecture 14 (and lab 6). **In this project, you are required to apply the reduction algorithm you learned in lab to all kernels where its use is reasonable.** Think about in which kernel you run a reduction-type calculation, combining several values into a single value where the operation is associative and commutative. As a hint, there is more than 1 kernel where reduction can be reasonably used.
