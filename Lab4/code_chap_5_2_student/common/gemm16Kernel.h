typedef unsigned short half;

#ifdef __cplusplus
extern "C" {
#endif
// 补齐函数声明，对应gemm/gemm_SRAM.mlu
void gemm16Kernel(...);
#ifdef __cplusplus
}
#endif
