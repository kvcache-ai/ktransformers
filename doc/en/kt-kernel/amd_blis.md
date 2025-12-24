
### USAGE
1. To use this feature, you should use MOE_INT8 method (i.e. `--kt-method MOE_INT8`)
2. !!! you should see the method in the below motivation section to  build and install the correct amd blis lib.
3. Before your install you should set `export CPUINFER_ENABLE_BLIS=ON` to enable
### Motivation

To accelerate the prefill speed of AMD. Reference the https://github.com/amd/blis repo. And the usage should add the LPGEMM support. See the docs here: https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Bhaskar_BLIS_Retreat_2024_AMD_LPGEMM_0.pdf
I reference this api guide for the code: https://docs.amd.com/r/en-US/57404-AOCL-user-guide/AOCL-BLAS?section=lpgemm-in-aocl-blas
To use lpgemm, see the doc here: 
https://www.amd.com/content/dam/amd/en/documents/developer/version-4-1-documents/aocl/aocl-4-1-user-guide.pdf
<img width="2134" height="1240" alt="Image" src="https://github.com/user-attachments/assets/d4008736-c1c7-422e-a747-155fc2eb4141" />
So, you just need to enable aocl_gemm add-on, examples are here:https://github.com/amd/blis/blob/master/docs/CMakeBuildSystem.md

<img width="2222" height="702" alt="Image" src="https://github.com/user-attachments/assets/bf924b69-e01d-460d-b4cd-122e77ec982d" />
You can see how to install it.



