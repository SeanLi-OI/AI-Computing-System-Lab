/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

/************************************************************************
 *
 *  @file cnrt.h
 *
 *  @brief Runtime APIs provide programmable interfaces for users to develop
 *  their-owned programs, which includes device management, context
 *  management, memory management of both sides (devices and hosts), etc.
 *
 **************************************************************************/

#ifndef __CNRT_H
#define __CNRT_H

#define CNRT_MAJOR_VERSION 4
#define CNRT_MINOR_VERSION 2
#define CNRT_PATCH_VERSION 0

#define CNRT_VERSION (CNRT_MAJOR_VERSION * 10000 + CNRT_MINOR_VERSION * 100 + CNRT_PATCH_VERSION)

/************************************************************************
 *  Include files
 ************************************************************************/
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif /*__cplusplus*/

/************************************************************************
 *  Definitions
 ************************************************************************/
/**< DLL exports controller. */
#if defined(WIN32) || defined(WINDOWS)
#ifdef USE_CNRT_DLL
#ifdef CNRT_DLL_EXPORTS
#define CNRT_DLL_API __declspec(dllexport)
#else /*CNRT_DLL_EXPORTS*/
#define CNRT_DLL_API __declspec(dllimport)
#endif /*CNRT_DLL_EXPORTS*/
#else
#define CNRT_DLL_API
#endif /*USE_CNRT_DLL*/
#else  /*WIN32 || WINDOWS*/
#define CNRT_DLL_API
#endif /*WIN32 || WINDOWS*/

/**< struct tailed */
#define CNRT_PARAM_END (void *)0xFFFFFFFF

/************************************************************************
 *  Data type declaration
 ************************************************************************/

#ifndef __CAMB_TYPES_H
#define __CAMB_TYPES_H
#if defined(WIN32) || defined(WINDOWS)
typedef unsigned __int64 u64_t;
typedef __int64 i64_t;
typedef unsigned __int32 u32_t;
typedef unsigned __int16 u16_t;
typedef unsigned __int8 u8_t;
typedef signed __int32 i32_t;
typedef signed __int16 i16_t;
typedef signed __int8 i8_t;

#else /*!WIN32 || WINDOWS*/

typedef uint64_t u64_t;
typedef int64_t i64_t;
typedef uint32_t u32_t;
typedef uint16_t u16_t;
typedef uint8_t u8_t;
typedef int32_t i32_t;
typedef int16_t i16_t;
typedef int8_t i8_t;

#endif /*WIN32||WINDOWS*/
#endif /*__CAMB_TYPES*/

#define CNRT_CHECK(statment)                                                  \
  do {                                                                        \
    int ret_code = (statment);                                                \
    if (ret_code != CNRT_RET_SUCCESS) {                                       \
      printf("[%s:%d] CNRT error, code: %d\n", __FILE__, __LINE__, ret_code); \
      exit(1);                                                                \
    }                                                                         \
  } while (false);

/**< Return values for CNRT API calls */
//! @brief An enum describes CNRT API return values.
/*! Function return values of CNRT API interfaces. */
typedef enum {
  CNRT_RET_SUCCESS = 0,
  /*!< The operation was successful. */
  CNRT_RET_WARNING_FAKE_DEVICE = 1,
  /*!< that operations issued previously have not completed yet. */
  CNRT_RET_ERR_NOT_READY = 632006,
  /*!< Use fake device currently. */
  CNRT_RET_ERR_INVALID = 632007,
  /*!< The supplied argument was invalid argument. */
  CNRT_RET_ERR_NOMEM = 632008,
  /*!< Insuffcient memory for the operation. */
  CNRT_RET_ERR_NODEV = 632009,
  /*!< No such device. */
  CNRT_RET_ERR_IO = 632010,
  /*!< I/O error. */
  CNRT_RET_ERR_SYS = 632011,
  /*!< System error. */
  CNRT_RET_ERR_ACCES = 632012,
  /*!< Permission denied. */
  CNRT_RET_ERR_FAULT = 632013,
  /*!< Bad address. */
  CNRT_RET_ERR_BUSY = 632014,
  /*!< Device or resource busy. */
  CNRT_RET_ERR_TIMEOUT = 632015,
  /*!< Time expired. */
  CNRT_RET_ERR_EXIST = 632016,
  /*!< Resource or file already exists. */
  CNRT_RET_ERR_NOSYS = 632017,
  /*!< Function not implemenmted. */
  CNRT_RET_ERR_AGAIN = 632018,
  /*!< try again later. */
  CNRT_RET_ERR_NORES = 632019,
  /*!< Out of resource. */
  CNRT_RET_ERR_UNSUPPORTED = 632020,
  /*!< Unsupported operation. */
  CNRT_RET_ERR_INVALID_POINTER = 632021,
  /*!< Invalid pointer. */
  CNRT_RET_ERR_NO_EXIST = 632022,
  /*!< Resource or file doesn't exist. */
  CNRT_RET_ERR_BROKEN = 632023,
  /*!< Data transmission is broken. */
  CNRT_RET_ERR_INIT = 632024,
  /*!< Uninitialized. */
  CNRT_RET_ERR_STREAM = 632025,
  /*!< Failure on Stream. */
  CNRT_RET_ERR_QUEUE = 632025,
  /*!< Failure on Queue. */
  CNRT_RET_ERR_OUT_RANGE = 632026,
  /*!< Number out of range. */
  CNRT_RET_ERR_MATH_OVERFLOW = 632027,
  /*!< Math result not representable. */
  CNRT_RET_ERR_FUNC_CALL = 632028,
  /*!< Failure to call runtime functions. */
  CNRT_RET_ERR_UNHANDLED = 632029,
  /*!< Unhandled error. */
  CNRT_RET_ERR_INVALID_TYPE = 632030,
  /*!< Invalid type. */
  CNRT_RET_ERR_INVALID_OP = 632031,
  /*!< Invalid operation. */
  CNRT_RET_ERR_MLU = 632032,
  /*!< MLU error. */
  CNRT_RET_ERR_ONCHIP_CORE = 632033,
  /*!< Onchip core error. */
  CNRT_RET_ERR_EVENT = 632034,
  /*!< Failure on event operation. */
  CNRT_RET_ERR_NOTIFIER = 632034,
  /*!< Failure on notifier operation. */
  CNRT_RET_ERR_RESHAPE = 632035,
  /*!< Failure on data reshape. */
  CNRT_RET_ERR_MEMCPY = 632036,
  /*!< Failure on memory copy. */
  CNRT_RET_ERR_ENCRYPT = 632037,
  /*!< Failure on encrypt. */
  CNRT_RET_ERR_INVALID_DATADESC = 632038,
  /*!< Invalid data descriptor. */
  CNRT_RET_ERR_MAP = 632039,
  /*!< Failure on map. */
  CNRT_RET_ERR_UNMAP = 632040,
  /*!< Failure on unmap. */
  CNRT_RET_ERR_CACHE = 632041,
  /*!< Failure on flush cache. */
  CNRT_RET_ERR_FIND_DEV_ADDR = 632042,
  /*!< Failure on find dev addr. */
  CNRT_RET_ERR_KERNEL_VERSION_TOO_HIGH = 632043,
  /*!< Kernel version too high, not supported. */
  CNRT_RET_ERR_UNKNOWN = 999991,
  /*!< Unknown error. */
  CNRT_RET_ERR_MAX
  /*!< The last one. */
} cnrtRet_t;

/**< Memory types available for allocator */
//! @brief An enum describes memory types available for allocator.
/*! Enumeration types, used to represent the memory types. */
typedef enum {
  CNRT_MEMTYPE_DEFAULT = 0,
  /*!< Host user space pagable memory. */
  CNRT_MEMTYPE_LOCKED,
  /*!< Host user space pinned memory. */
  CNRT_MEMTYPE_DEV
  /*!< Device memory. */
} cnrtMemType_t;

/**< Malloc types available for cnrtMallocBufferEx. */
//! @brief An enum describes malloc types available for cnrtMallocBufferEx.
/*! Internal enum. */
typedef enum { CNRT_MALLOC_EX_PARALLEL_FRAMEBUFFER = 1 } cnrtMallocExType_t;

/**< Execution modes of tasks on MLU. */
//! @brief An enum execution modes of tasks on MLU.
/*! The number of cores running on the Function of a device. */
typedef enum {
  CNRT_FUNC_TYPE_BLOCK = 1,
  /*!< Use 1 core. */
  CNRT_FUNC_TYPE_BLOCK0 = CNRT_FUNC_TYPE_BLOCK,
  /*!< Use IP core 0. */
  CNRT_FUNC_TYPE_BLOCK1 = CNRT_FUNC_TYPE_BLOCK0 + 1,
  /*!< Use IP heterogeneous core 1. */
  CNRT_FUNC_TYPE_UNION1 = 4,
  /*!< Use 4 cores. */
  CNRT_FUNC_TYPE_UNION2 = 8,
  /*!< Use 8 cores. */
  CNRT_FUNC_TYPE_UNION4 = 16,
  /*!< Use 16 cores. */
  CNRT_FUNC_TYPE_UNION8 = 32,
  /*!< Use 32 cores. */
  CNRT_FUNC_TYPE_MUTABLE = -1,
  /*!< Flexible mode. */
  CNRT_JOB_TYPE_BLOCK = CNRT_FUNC_TYPE_BLOCK,
  /*!< Use 1 core. */
  CNRT_JOB_TYPE_UNION1 = CNRT_FUNC_TYPE_UNION1,
  /*!< Use 4 cores. */
  CNRT_JOB_TYPE_UNION2 = CNRT_FUNC_TYPE_UNION2,
  /*!< Use 8 cores. */
  CNRT_JOB_TYPE_UNION4 = CNRT_FUNC_TYPE_UNION4,
  /*!< Use 16 cores. */
} cnrtFunctionType_t,
    cnrtJobType_t;

/**< DDR Channel for tasks used on MLU. */
//! @brief An enum describe DDR Channel for tasks used on MLU..
/*! Used to represent Channel types. */
typedef enum {
  CNRT_CHANNEL_TYPE_DUPLICATE = -2,
  /*!< Duplicate data on DDR channels, used in runtime context. */
  CNRT_CHANNEL_TYPE_NONE = -1,
  /*!< Use random channel. */
  CNRT_CHANNEL_TYPE_0 = 0,
  /*!< Use DDR channel 0. */
  CNRT_CHANNEL_TYPE_1,
  /*!< Use DDR channel 1. */
  CNRT_CHANNEL_TYPE_2,
  /*!< Use DDR channel 2. */
  CNRT_CHANNEL_TYPE_3
  /*!< Use DDR channel 3. */
} cnrtChannelType_t;

/**< Direction of data transmission. */
//! @brief An enum describes direction of data transmission.
/*! Direction of data transmission. */
typedef enum {
  CNRT_MEM_TRANS_DIR_HOST2DEV = 0,
  /*!< From host to device. */
  CNRT_MEM_TRANS_DIR_DEV2DEV,
  /*!< From device to device, in one device internally */
  CNRT_MEM_TRANS_DIR_DEV2HOST,
  /*!< From device to host */
  CNRT_MEM_TRANS_DIR_HOST2HOST,
  /*!< From host to host, not supported yet */
  CNRT_MEM_TRANS_DIR_PEER2PEER,
  /*!< From device to device,between two peerable devices */
  CNRT_MEM_TRANS_DIR_NODIR,
  /*!< no direction for init */
} cnrtMemTransDir_t;

/**< Action about cache. */
//! @brief An enum describes Action about cache.
/*! Action about cache. */
typedef enum { CNRT_FLUSH_CACHE = 1, CNRT_INVALID_CACHE = 2 } cnrtCacheOps_t;

/**< Parameter for function call */
/*!
 *  @struct cnrtDim3_t
 *  @brief A struct describes parameter for function call.
 *
 *  Dimension of task execution */
typedef struct {
  unsigned int x; /*!< x aixs */
  unsigned int y; /*!< y aixs */
  unsigned int z; /*!< z aixs */
} cnrtDim3_t;

/**< Parameter for invoke function call*/
/*!
 *  @struct cnrtInvokeFuncParam_t
 *  @brief A struct.
 *
 *  Parameters of the interface cnrtInvokeFunction (), which need to be invoked by the user. */
typedef struct {
  int *data_parallelism;  /*!< data parallelism*/
  unsigned int *affinity; /*!< affinity*/
  void *end;              /*!< end of struct*/
} cnrtInvokeFuncParam_t;

/**< Type of cnrtInvokeParam. */
//! @brief An enum describes type of cnrtInvokeParam.
/*! Type of cnrtInvokeParam. */
typedef enum {
  CNRT_INVOKE_PARAM_TYPE_0 = 0,
  /*!< type 0 cnrtInvokeParam */
} cnrtInvokeParamType_t;

/**< Parameter for function call */
/*!
 *  @struct cnrtClusterAffinity_t
 *  @brief A struct describes parameter for function call.
 *
 *  cluster of task execution */
typedef struct { unsigned int *affinity; /*!< affinity*/ } cnrtClusterAffinity_t;

/**< Parameter for function call */
/*!
 *  @struct cnrtInvokeParam_t
 *  @brief A struct describes parameter for function call.
 *
 * Parameters of the interface cnrtInvokeFunction_V3 (), cnrtInvokeFunctionExtra_V2 () and
 * cnrtInvokeKernel_V3 () which need
 * to be invoked by the user */
typedef struct {
  cnrtInvokeParamType_t invoke_param_type;
  /*!< Invoke param type */
  cnrtClusterAffinity_t cluster_affinity;
  /*!< Invoke cluster affinity */
} cnrtInvokeParam_t;

/**< Data type and data order*/
//! @brief An enum.
/*! Data types */
typedef enum cnrtDataType {
  CNRT_INVALID = 0x0,
  /*!< Invalid data */
  CNRT_FLOAT16 = 0x12,
  /*!< 16-bit floating-point data */
  CNRT_FLOAT32 = 0x13,
  /*!< 32-bit floating-point data */
  CNRT_FLOAT64 = 0x14,
  /*!< 64-bit floating-point data */

  CNRT_INT4 = 0x20, /* new element*/

  CNRT_INT8 = 0x21,
  /*!< 8-bit integer */
  CNRT_INT16 = 0x22,
  /*!< 16-bit integer */
  CNRT_INT32 = 0x23,
  /*!< 32-bit integer */
  CNRT_INT64 = 0x24,
  /*!< 64-bit integer */
  CNRT_AUTO = 0x25,
  /*!< automatic bit-width integer, change between int8 int16 etc. */

  CNRT_UINT8 = 0x31,
  /*!< 8-bit unsigned integer */
  CNRT_UINT16 = 0x32,
  /*!< 16-bit unsigned integer */
  CNRT_UINT32 = 0x33,
  /*!< 32-bit unsigned integer */
  CNRT_FIX8 = 0x41,
  /*!< 8-bit fixed-point data */
  CNRT_QUANT8 = 0x51,
  /*!< 8-bit data */
  CNRT_BOOL = 0x61,
  /*!< Boolean type */
} cnrtDataType_t;

//! @brief An enum.
/*! Used to represent the format of data placement.
 * Data can be divided into at least four dimensions.
 * Take pictures as an example, the order of placement can be:
 * the number of pictures, the number of picture Channels,
 * the height of the pictures, and the width of pictures (NCHW).
 */
typedef enum cnrtDimOrder {
  CNRT_NCHW = 0x0123,
  /*!< Placed by the NCHW dimension orders */
  CNRT_NHWC = 0x0231,
  /*!< Placed by the NHWC dimension orders */
  CNRT_HWCN = 0x2310,
  /*!< Placed by the HWCN dimension orders */
  CNRT_TNC = 0x401,
  /*!< Placed by the TNC dimension orders(RNN exclusive) */
  CNRT_NTC = 0x041,
  /*!< Placed by the NTC dimension orders(RNN exclusive) */
  CNRT_NCDHW = 0x01523,
  /*!< Placed by the NCHW dimension orders */
  CNRT_NDHWC = 0x05231,
  /*!< Placed by the NHWC dimension orders */
  CNRT_DHWCN = 0x52310,
  /*!< Placed by the HWCN dimension orders */
} cnrtDimOrder_t;

//! @brief An enum.
/*! Context types */
typedef enum cnrtRuntimeContextInfo {
  CNRT_RT_CTX_FUNCTION = 1,
  /*!< Computation unit */
  CNRT_RT_CTX_DEV_ORDINAL = 2,
  /*!< Device ordinal */
  CNRT_RT_CTX_CORE_NUMBER = 3,
  /*!< Core number set by compile time */
  CNRT_RT_CTX_MODEL_PARALLEL = 4,
  /*!< Degree of model parallelism */
  CNRT_RT_CTX_CHANNEL = 5,
  /*!< Channel of device memory */
  CNRT_RT_CTX_MAX_BATCH_NUM = 6,
  /*!< Maximum batch number that cnrtInvokeRuntimeContextBatch could take */
} cnrtRuntimeContextInfo_t;

//! @brief An enum.
/*! Device types */
typedef enum cnrtCoreVersion {
  CNRT_1H8 = 0,
  /*!< 1H8 hardware */
  CNRT_1H16 = 1,
  /*!< 1H16 hardware */
  CNRT_1H8MINI = 4,
  /*!< 1H8MINI hardware */
  CNRT_MLU100 = 3,
  /*!< MLU100 hardware */
  CNRT_MLU270 = 5,
  /*!< MLU270 hardware */
  CNRT_MLU220 = 6,
  /*!< MLU220 hardware */
} cnrtCoreVersion_t;

/**< Parameter for cnrtGetDeviceInfo function call*/
/*!
 *  @struct cnrtDeviceInfo_t
 *  @brief A struct.
 *
 *  Parameters of the interface cnrtGetDeviceInfo(), for get the device info. */
typedef struct {
  char device_name[64];           /*!< device name */
  cnrtCoreVersion_t core_version; /*!< device core version */
  int core_num;                   /*!< device core num */
} cnrtDeviceInfo_t;

/**< Device affinity information */
/*!
 *  @struct cnrtDeviceAffinity_t
 *  @brief A struct.
 *
 *  A struct describing the device affinity */
typedef struct {
  uint32_t cpu_count; /*!< The number of CPUs having an affinity with the specified devices */
  uint32_t cpu_affinity_bitmap[1024]; /*!< Obtain the affinity bitmask of the specified card */
} cnrtDeviceAffinity_t;

/**< topology relationship */
//! @brief An enum.
/*! Topology struct */
typedef enum {
  CNRT_TOPO_SELF = 0,
  CNRT_TOPO_INTERNAL = 1,
  /*!< devices that are on the same board */
  CNRT_TOPO_SINGLE = 2,
  /*!< all devices that only need traverse a single PCIe switch */
  CNRT_TOPO_MULTIPLE = 3,
  /*!< all devices that need not traverse a host bridge */
  CNRT_TOPO_HOST_BRIDGE = 4,
  /*!< all devices that are connected to the same host bridge */
  CNRT_TOPO_CPU = 5,
  /*!< all devices that are connected to the same CPU */
  CNRT_TOPO_SYSTEM = 6
  /*!< all device in the system */
} cnrtTopologyRelationshipEnum_t;

struct cnrtQuantizedParam;

typedef struct cnrtQuantizedParam *cnrtQuantizedParam_t;
/**< Model and function */
/*!
 *  @struct cnrtModel
 *  @brief A struct.
 *
 *  Semi-internal struct. A struct describing Model */
struct cnrtModel;
/*! A pointer which points to the struct describing Model */
typedef struct cnrtModel *cnrtModel_t;

/*!
 *  @struct cnrtFunction
 *  @brief A struct.
 *
 *  Semi-internal struct. A struct describing Function */
struct cnrtFunction;
/*! A pointer which points to the struct describing Function */
typedef struct cnrtFunction *cnrtFunction_t;

/**< Parameter descriptor */

/*!
 *  @struct cnrtParamDesc
 *  @brief A struct that describes the attribute(shape, order, datatype)
 *  of input or output parameter.
 *
 *  You can specify the attribute of input and output parameters by cnrtParamDesc
 *  and pass them to cnrtInvokeFunction_V3 or cnrtInvokeRuntimeContext_V2. */
struct cnrtParamDesc;
/*! A pointer which points to cnrtParamDesc */
typedef struct cnrtParamDesc *cnrtParamDesc_t;
/*! ``cnrtParamDesc_t`` is a second rank pointer to ``cnrtParamDesc`` which is a
     structure holding the description of IO param. */
typedef struct cnrtParamDesc **cnrtParamDescArray_t;

/*!
 *  @struct cnrtQueue
 *  @brief A struct.
 *
 *  Semi-internal struct. A struct describing Queue */
struct cnrtQueue;
/*! A pointer which points to the struct describing Queue */
typedef struct cnrtQueue *cnrtQueue_t;

/*!
 *  @struct cnrtNotifier
 *  @brief A struct.
 *
 *  Semi-internal struct. A struct describing Notifier */
struct cnrtNotifier;
/*! A pointer which points to the struct describing Notifier */
typedef struct cnrtNotifier *cnrtNotifier_t;

/*!
 *  @struct cnrtRuntimeContext
 *  @brief A struct.
 *
 *  A struct describing runtime context */
struct cnrtRuntimeContext;
/*! A pointer which points to the struct describing runtime context */
typedef struct cnrtRuntimeContext *cnrtRuntimeContext_t;

typedef u64_t cnrtDev_t;

/**< Compiler */
/*!
 *  @struct cnrtKernelParamsBuffer
 *  @brief A struct.
 *
 *  Internal struct.  */
struct cnrtKernelParamsBuffer;
typedef struct cnrtKernelParamsBuffer *cnrtKernelParamsBuffer_t;

struct cnrtPluginOpDimInfo;
typedef struct cnrtPluginOpDimInfo *cnrtPluginOpDimInfo_t;

typedef struct cnrtKernelParamsBuffer {
  void *host_ptr;
  unsigned int max_param;
  unsigned int cur_param;

  // for plugin op
  // mark the position of kernel input/output/static ptr in param
  int *input_index;
  int num_input;
  int *output_index;
  int num_output;
  int *static_index;
  int num_static;

  // for plugin op
  // mark the postion of tensor dim info in param.
  cnrtPluginOpDimInfo_t dim_info;
  int num_dim_info;
} * cnrtKernelParamsBuffer_t;

/*!
 * @struct cnrtKernelInitParam.
 * @brief A struct.
 *
 * A struct describing kernel init param. */
struct cnrtKernelInitParam;
typedef struct cnrtKernelInitParam *cnrtKernelInitParam_t;

/************************************************************************
 * Function prototype declaration
 ************************************************************************/

/************************************************************************
 * Error handling
 ************************************************************************/

/**
 * @brief Return string pointer that describes
 *     the error code passed in the argument errCode.
 *
 * The function returns a read only string that is corresponding
 * to the argument @p errcode.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  err_code[in] the error code was returned by previous function call.
 * @retval a pointer that points to a constant string.
 */
extern CNRT_DLL_API const char *cnrtGetErrorStr(cnrtRet_t err_code);

/**
 * @brief Get the error code set by any runtime calls.
 *     Its value is meaningful only when the return value indicating an error.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @retval error code of the last call of runtime functions.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetLastErr(void);

/*************************************************************************
 * Initialization and destroy
 *************************************************************************/

/**
 * @brief Initialize runtime environment in current process space.
 *
 * Initializes this API must be called before any other runtime API calls.
 *
 * To initialize a fake device:
 *
 * 1. Call the cnrtInit API and set the flags[in] to 1.
 *
 *    cnrtInit(1);
 *
 * 2. Declare cnrtDev_t
 *
 *    cnrtDev_t dev;
 *
 * 3. Call the cnrtGetDeviceHandle API and set ordinal[in] to -1.
 *
 *    cnrtGetDeviceHandle(&dev, -1);
 *
 * 4. Call the cnrtSetCurrentDevice API.
 *
 *    cnrtSetCurrentDevice(dev);
 *
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  flags[in] Reserved for further use, pass 0 as well. If you
           set the value of this parameter to 0, the real device is
                   initialized. If you set the value of this parameter to 1,
                   the fake device is initialized.
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInit(unsigned int flags);

/**
 * @brief Destroy everything that allocated by runtime API calls.
 *
 * This API should be called after any other runtime API calls.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @retval void (None).
 */
extern CNRT_DLL_API void cnrtDestroy(void);

/******************************************************************************
 * Version and revision
 ******************************************************************************/

/**
 * @brief Return the version of the CNRT software.
 *
 * Higher version usually offers more features provided by this library.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  ver[out] pointer to retrieve the version.
 * @retval unsigned int for version number.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetVersion(unsigned int *ver);

/******************************************************************************
 * Device managment
 ******************************************************************************/

/**
 * @brief Get the device handle by a given device ordinal.
 *
 *  The function returns the device handle given a specific device ordinal.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  pdev[out] pointer to retrieve the device handle.
 * @param  ordinal[in] the device ordinal to get the device handle.
 * @note   The value of the ordinal parameter should be in the range
           [0~cnrtGetDeviceCount() - 1]. The value -1 represents a fake device.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */

extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceHandle(cnrtDev_t *pdev, int ordinal);

/**
 * @brief Set the device handle for current thread execution context.
 *
 *  It implies that any subsequent runtime API calls are for this device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  dev[in] the device handle.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentDevice(cnrtDev_t dev);

/**
 * @brief Get the cnrtDevice handle from current thread execution context.
 *
 * The handle has been set by calling cnrtSetCurrentDevice().
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  pdev[out] pointer to retrieve the device handle.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentDevice(cnrtDev_t *pdev);

/**
 * @brief Get the number of MLU devices in the system.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param  dev_num[out] pointer to retrieve the number of devices.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceCount(unsigned int *dev_num);

/**
 * @brief Get the information about the specified device
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param info[out] infor for the specified device.
 * @param device_ordinal[in] device ordinal to get device info for.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceInfo(cnrtDeviceInfo_t *info, int device_ordinal);

/**
 * @brief  Wait for the device to complete precedent tasks.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSyncDevice(void);

/******************************************************************************
 * Queue management
 ******************************************************************************/

/**
 * @brief Create a new queue after calling this function,
 *        it works in asynchronous mode by default.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pQueue[out] pointer to retrieve the new created Queue handle.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 * @attention Queue numbers should be not greater than 4094 on MLU270,
 *            not greater than 1024 on MLU100.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQueue(cnrtQueue_t *pQueue);

/**
 * @brief Destroy a queue created by calling cnrtCreateQueue.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyQueue(cnrtQueue_t queue);

/**
 * @brief Function should be blocked until all precedent tasks in the queue are completed.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @retval CNRT_RET_SUCCESS if success, otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSyncQueue(cnrtQueue_t queue);

/*********************************************************************************
 * Notifier, only MLU100 support
 *********************************************************************************/

/**
 * @brief Create a notifier corresponding to the current device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param notifier[out] point to an notifier handle to retrieve newly created notifier.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateNotifier(cnrtNotifier_t *notifier);

/**
 * @brief Destroy a notifier that was created by calling cnrtCreateNotifier.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param notifier[in] notifier handle to be destroyed.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyNotifier(cnrtNotifier_t *notifier);

/**
 * @brief Wait notifier which has been placed to queue by calling cnrtPlaceNotifier
 *        util it is in the signaled state or exceeds the time-out interval.
 *        This function will block CPU thread.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param notifier[in] event handle created by calling cnrtCreateNotifier.
 * @retval CNRT_RET_SUCCESS if success.
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtWaitNotifier(cnrtNotifier_t notifier);

/**
 * @brief Query the status notifier which has been placed to queue by calling cnrtPlaceNotifier.
 *        This function will not block CPU thread.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param notifier[in] notifier handle created by calling cnrtCreateNotifier.
 *
 * @retval CNRT_RET_SUCCESS if notification instruction has been executed,
 *         CNRT_RET_ERR_BUSY if the preceding tasks is still in progress,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryNotifier(cnrtNotifier_t notifier);

/**
 * @brief Place a notifier in specified queue. This function will not block the CPU thread.
 *        All computation tasks submitted to the queue will wait until event reports
 *        completion before starting execution.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param notifier[in] signal handle created by calling cnrtCreateNotifier.
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtPlaceNotifier(cnrtNotifier_t notifier, cnrtQueue_t queue);

/**
 * @brief Make the specified queue wait for a notifier. This function is designed for
 *        cross queue synchronization.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param notifier[in] signal handle created by calling cnrtCreateNotifier.
 * @param queue[in] queue handle created by calling cnrtCreateQueue or cnrtCreateQueueEx.
 * @param flag[in] flags control operation.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueueWaitNotifier(cnrtNotifier_t notifier,
                                                    cnrtQueue_t queue,
                                                    unsigned int flag);

/**
 * @brief Get duration time of two makers.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param start[in] notifier handle created by calling cnrtCreateNotifier.
 * @param end[in] notifier handle created by calling cnrtCreateNotifier.
 * @param us[out] duration time between start and end.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtNotifierDuration(cnrtNotifier_t start,
                                                   cnrtNotifier_t end,
                                                   float *us);

/**< Compiler */
/*********************************************************************************
 * Execution control
 *********************************************************************************/

/**
 * @brief Get a parameter buffer for cnrtInvokeKernel.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param params[in] pointer to a param buffer
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetKernelParamsBuffer(cnrtKernelParamsBuffer_t *params);

/**
 * @brief Copy Parambuffer from src_params_buf to dst_params_buf
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dst_params_buf[in] pointer to an allocated param buffer
 * @param src_params_buf[in] pointer to an allocated param buffer
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCopyKernelParamsBuffer(cnrtKernelParamsBuffer_t dst_params_buf,
                                                         cnrtKernelParamsBuffer_t src_params_buf);

/**
 * @brief Add a parameter to a specific parameter buffer.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param params[in] destination parameter buffer
 * @param data[in] pointer to host memory
 * @param bytes[in] size in bytes
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferAddParam(cnrtKernelParamsBuffer_t params,
                                                             void *data,
                                                             size_t bytes);

/**
 * @brief Add a InputPtr place holder to a specific parameter buffer.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param params[in] destination parameter buffer
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferMarkInput(cnrtKernelParamsBuffer_t params);

/**
 * @brief Add a OutputPtr place holder to a specific parameter buffer.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param params[in] destination parameter buffer
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferMarkOutput(cnrtKernelParamsBuffer_t params);

/**
 * @brief Add a StaticPtr place holder to a specific parameter buffer.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param params[in] destination parameter buffer
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtKernelParamsBufferMarkStatic(cnrtKernelParamsBuffer_t params);

/**
 * @brief Destroy a parameter buffer returned by cnrtGetKernelParamsBuffer.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param params[in] pointer to a param buffer
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyKernelParamsBuffer(cnrtKernelParamsBuffer_t params);

/**
 * @brief Invoke a kernel written in Bang with given params on MLU.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param function[in] point to the MLU function.
 * @param dim[in] how many grid dimensions.
 * @param params[in] point to arguments.
 * @param func_type[in] function type. @see cnrtFunctionType_t.
 * @param queue[in] queue associated to the function call.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeKernel_V2(const void *function,
                                                  cnrtDim3_t dim,
                                                  cnrtKernelParamsBuffer_t params,
                                                  cnrtFunctionType_t func_type,
                                                  cnrtQueue_t queue);

/**
 * @brief Create a kernel init param.
 *
 *  **Supports both MLU220 and MLU270**
 *
 * @param init_param[in] pointer to cnrtKernelInitParam_t.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateKernelInitParam(cnrtKernelInitParam_t *init_param);
/**
 * @brief Initialize a kernel memory, the kernel is written in Bang.
 *
 *  **Supports both MLU220 and MLU270**
 *
 * notice: cnrtInitKernelMemory should be called before cnrtInvokeKernel_V3 and after
 * cnrtCreateKernelInitParam.
 *
 * @param function[in] pointer to MLU function.
 * @param init_param[in] kernel init param created by cnrtCreateKernelInitParam.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInitKernelMemory(const void *function,
                                                   cnrtKernelInitParam_t init_param);
/**
 * @brief Invoke a kernel written by Bang with given params on MLU.
 *
 *  **Supports both MLU220 and MLU270**
 *
 * notice: cnrtInvokeKernel_V3 should be called after cnrtInitKernelMemory. For a bang function, you
 * should call cnrtCreateKernelInitParam and cnrtInitKernelMemory only once, and call
 * cnrtInvokeKernel_V3 many times if you need invoke a bang function multi-times.
 *
 * @param function[in] pointer to MLU function.
 * @param init_param[in] kernel init param created by cnrtCreateKernelInitParam and used by
 * cnrtInitKernelMemory.
 * @param dim[in] how many grid dimensions.
 * @param params[in] point to arguments.
 * @param func_type[in] function type. @see cnrtFunctionType_t.
 * @param queue[in] queue associated to the function call.
 * @param extra_param[in] pointer to cnrtInvokeParam_t as extra param.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeKernel_V3(const void *function,
                                                  cnrtKernelInitParam_t init_param,
                                                  cnrtDim3_t dim,
                                                  cnrtKernelParamsBuffer_t params,
                                                  cnrtFunctionType_t func_type,
                                                  cnrtQueue_t queue,
                                                  void *extra_param);
/**
 * @brief destroy Bang-kernel init param and memory.
 *
 *  **Supports both MLU220 and MLU270**
 *
 * @param init_param[in] kernel init param created by cnrtCreateKernelInitParam, used by
 * cnrtInitKernelMemory and cnrtInvokeKernel_V3.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyKernelInitParamAndMemory(cnrtKernelInitParam_t param);

/*********************************************************************************
 * Model load and Function call
 *********************************************************************************/

/**
 * @brief Load a model from a given model file.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pmodel[out] point to a cnrtModel_t.
 * @param fname[in]  file name of a cambricon model.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModel(cnrtModel_t *pmodel, const char *fname);

/**
 * @brief Load a model from memory
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pmodel[out] pointer to a cnrtModel_t.
 * @param ptr[in] memory pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtLoadModelFromMem(cnrtModel_t *pmodel, char *ptr);

/**
 * @brief Unload a model.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] point to a cnrtModel_t.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtUnloadModel(cnrtModel_t model);

/**
 * @brief  Get actual size of model in offline file.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param fname[in] file name of a cambricon model.
 * @param size[out] pointer to model's actual size.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelSize(const char *fname, int *size);

/**
 * @brief  Query model's core version, 1H8 or 1H16.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] pointer to a loaded model.
 * @param coreVersion[out] pointer to model's core version.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryCoreVersion(cnrtModel_t model,
                                                   cnrtCoreVersion_t *coreVersion);

/**
 * @brief  Query model's parallelism, which means the core number
 * involved to compute this model.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] point to a loaded model.
 * @param modelParallelism[out] pointer to model's parallelism.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelParallelism(cnrtModel_t model, int *modelParallelism);

/**
 * @brief  Query model's stack size, which is the biggest stack size(MB)
 * in all the kernels in the model.
 *
 * Deprecated. This interface will be deleted in the next version and
 * cnrtQueryModelLocalMemSize is recommended to use.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] point to a loaded model.
 * @param size[out] pointer to the stack size.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelStackSize(cnrtModel_t model, uint64_t *stack_size);

/**
 * @brief  Query model's local memory size, which is the biggest local memory size(MB)
 * in all the kernels in the model.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] point to a loaded model.
 * @param local_mem_size[out] pointer to the local memory size.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtQueryModelLocalMemSize(cnrtModel_t model,
                                                         uint64_t *local_mem_size);

/**
 * @brief Get function number of a given model
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] pointer of a cambricon model
 * @param func_num[out] pointer to function number
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetFunctionNumber(cnrtModel_t model, int *func_num);

/**
 * @brief Extract the symbol from the given model if symbol exists,
 *        otherwise error code will be returned.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param function[out] point to a cnrtFunction_t.
 * @param model[in]  point to a loaded model.
 * @param symbol[in] symbol name.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtExtractFunction(cnrtFunction_t *pfunction,
                                                  cnrtModel_t model,
                                                  const char *symbol);

/**
 * @brief Create a MLU function.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param function[in] pointer of cnrtFunction_t.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateFunction(cnrtFunction_t *pfunction);

/**
 * @brief Destroy a function.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyFunction(cnrtFunction_t function);

/**
 * @brief get index of paramdesc by name from a function.
 *
 *  **Supports only MLU270**
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @param name[in] point to a name setted to tensor before compile.
 * @param index[out] point to a index, will return right index of param_desc while name match.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamIndexByName(cnrtFunction_t func,
                                                      const char *name,
                                                      int *index);

/**
 * @brief get support shape dim_num by name from a function.
 *
 *  **Supports only MLU270**
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @param name[in] point to a name setted to tensor before compile.
 * @param dim_num[out] point to a int, will return right dim num of param_desc while name match.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamDimNumberByName(cnrtFunction_t func,
                                                                   const char *name,
                                                                   int *dim_num);

/**
 * @brief get support shape value by name from a function.
 *
 *  **Supports only MLU270**
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @param name[in] point to a name setted to tensor before compile.
 * @param dim_shape[out] point to dim_num int values, will return right shape of param_desc while
 * name match.
 *        value will be -1 when dim is variable.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamShapeByName(cnrtFunction_t func,
                                                               const char *name,
                                                               int *dim_shape);

/**
 * @brief get support datatype by name from a function.
 *
 *  **Supports only MLU270**
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @param name[in] point to a name setted to tensor before compile.
 * @param dtype[out] point to a cnrt datatype, will return right datatype of param_desc while name
 * match.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamDataTypeByName(cnrtFunction_t func,
                                                                  const char *name,
                                                                  cnrtDataType_t *dtype);

/**
 * @brief get support dim_order by name from a function.
 *
 *  **Supports only MLU270**
 *
 * @param function[in] point to a function generated by cnrtExtractFunction.
 * @param name[in] point to a name setted to tensor before compile.
 * @param dorder[out] point to a cnrt order, will return right order of param_desc while name match.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetSupportedParamLayoutByName(cnrtFunction_t func,
                                                                const char *name,
                                                                cnrtDimOrder_t *dorder);

/**
 * @brief Generate a copy of source MLU function. src and dst function share the
 *        same kernel on host, but they have different device space, so model
 *        data(include instruction) is doubled on device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param src[in] Pointer to a source MLU function
 * @param dst[out] Pointer to a destination MLU function pointer
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCopyFunction(cnrtFunction_t *dst, cnrtFunction_t src);

/*********************************************************************************
 * Memory management
 *********************************************************************************/

/**
 * @brief Allocate nByte bytes and place a pointer to pointer
 *        in pPtr to the allocated host memory. If bytes is 0, then
 *        cnrtMallocHost returns either NULL, or a unique pointer value
 *        that can later be passed to cnrtFreeHost.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pPtr[out]  a pointer to pointer for retrieving allocated host memory.
 * @param bytes[in] number bytes of memory to be allocated.
 * @param type[in]   memory type to be allocated,
 *                   @see CNRT_HOST_MEMORY_TYPE_LOCK and CNRT_HOST_MEMORY_TYPE_MAPPED.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocHost(void **pPtr, size_t bytes, cnrtMemType_t type);

/**
 * @brief Free the memory space pointed by ptr, which must be
 *        returned by a previous call of cnrtMallocHost.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param ptr[in]  point to the address of memory to be free.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFreeHost(void *ptr);

/**
 * @brief Allocate memory on MLU device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pPtr[out] a pointer to pointer for retrieving allocated device memory.
 * @param bytes[in] allocate size.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMalloc(void **pPtr, size_t bytes);

/**
 * @brief Allocate memory on MLU device. For P2P.
 *
 *  **Supports only MLU100**
 *
 * @param pPtr[out] a pointer to pointer for retrieving allocated device memory.
 * @param bytes[in] allocate size.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocFrameBuffer(void **pPtr, size_t bytes);

/**
 * @brief Allocate memory on MLU device, for extension
 *
 *  **Supports only MLU100**
 *
 * @param pPtr[out] a pointer to pointer for retrieving allocated device memory.
 * @param param[in] parameter buffer allocated by cnrtAllocParam
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMallocBufferEx(void **pPtr, void *param);

/**
 * @brief Deallocate MLU device Memory.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param ptr[in] point to the memory to be free.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFree(void *ptr);

/**
 * @brief Deallocate MLU multiple device memory addresses allocated
 *        by cnrtMallocBatchByDescArray, cnrtMallocByDescArray.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param ptr[in] a pointer array.
 * @param length[in] array length.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFreeArray(void **ptr, int length);

/**
 * @brief Map device addr returned by a previous call to cnrtMalloc
 *        into host addr in userspace.
 *
 *  **supports only MLU220_ARM**
 *
 * @param host_ptr[out] maped address of host.
 * @param dev_ptr[in]  address of device.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMap(void **host_ptr, void *dev_ptr);

/**
 * @brief Unmap the memory space pointed by host_ptr, which must
 *        be returned by a previous call to cnrtMap.
 *
 *  **supports only MLU220_ARM**
 *
 * @param host_ptr[in] point to the memory to be free.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtUnmap(void *host_ptr);

/**
 * @brief Get device address according to mappped_host_ptr
 *
 * @param dev_ptr[out] address of device.
 * @param mappped_host_ptr[in] mapped address of host.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtFindDevAddrByMappedAddr(void *mappped_host_ptr, void **dev_ptr);

/**
 * @brief Take an action in cache
 *
 *  **supports only MLU220_ARM**
 *
 * @param host_ptr[in] maped address of host.
 * @param opr[in] action about in cache.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCacheOperation(void *host_ptr, cnrtCacheOps_t opr);

/**
 * @brief Copy data from src address to dst address. The copy direction
 *        is specified by input parameter dir. The copy operation is
 *        always performed on current device which is set by cnrtSetCurrentDevice.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dst[in] destination address.
 * @param src[in] source address.
 * @param bytes[in] number of bytes to be copied.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 *                      CNRT_MEM_TRANS_DIR_HOST2HOST,
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemcpy(void *dst, void *src, size_t bytes, cnrtMemTransDir_t dir);

/**
 * @brief Aysnchronous copy data from src address to dst address. The copy direction
 *        is specified by input parameter dir. The copy operation is
 *        always performed on current device which is set by cnrtSetCurrentDevice.
 *
 *  **Supports only MLU270**
 *
 * @param dest[in] destination address.
 * @param src[in] source address.
 * @param bytes[in] number of bytes to be copied.
 * @param queue[in] queue handle created by calling cnrtCreateQueue.
 * @param dir[in] direction of transfer.
 *                @see  CNRT_MEM_TRANS_DIR_HOST2DEV,
 *                      CNRT_MEM_TRANS_DIR_DEV2HOST,
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtMemcpyAsync(void *dest, void *src, size_t bytes, cnrtQueue_t queue, cnrtMemTransDir_t dir);

/**
 * @brief Fill the bytes of the device memory space
 *        pointed by devPtr with the constant value c.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param ptr[in] device memory address.
 * @param c[in] value to be filled.
 * @param bytes[in] number of bytes to be filled.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtMemset(void *ptr, int c, size_t bytes);

/**
 * @brief set MLU stack space memory to stack_size(MB).
 *
 * Deprecated. This interface will be deleted in the next version and
 * cnrtSetLocalMem is recommended to use.
 *
 *  **Supports both MLU100 and MLU270.**
 *
 * @param stacksize[in] the size of MLU stack space memory will be set.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise CNRT_RET_ERR_MLU is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetStackMem(unsigned int stacksize);

/**
 * @brief get MLU stack space memory to stack_size(MB).
 *
 * Deprecated. This interface will be deleted in the next version and
 * cnrtGetLocalMem is recommended to use.
 *
 *  **Supports both MLU100 and MLU270.**
 *
 * @param pStacksize[out] the size of MLU stack space memory will be get.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise CNRT_RET_ERR_MLU is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetStackMem(unsigned int *pStacksize);

/**
 * @brief set MLU local memory space memory(MB).
 *
 *  **Supports both MLU100 and MLU270.**
 *
 * @param local_mem_size[in] the size of MLU local memory space memory will be set.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise CNRT_RET_ERR_MLU is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetLocalMem(unsigned int local_mem_size);

/**
 * @brief get MLU local memory space(MB).
 *
 *  **Supports both MLU100 and MLU270.**
 *
 * @param pLocalsize[out] the size of MLU local memory space will be get.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise CNRT_RET_ERR_MLU is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetLocalMem(unsigned int *pLocalsize);

/**
 * @brief get max memory used of function
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param function[in] point to the MLU function.
 * @param pMemused[out] return value.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetFunctionMemUsed(cnrtFunction_t function, int64_t *pMemused);

/**
 * @brief get max memory used of model
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param model[in] point to the model.
 * @param pMemused[out] return value.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelMemUsed(cnrtModel_t model, int64_t *pMemused);

/*********************************************************************************
 * Channel control, only MLU100 support
 *********************************************************************************/

/**
 * @brief Set memory and computation channel on current MLU device. Once
 *        a channel is configured, all memory allocation(eg. cnrtMalloc)
 *        will be performed on this channel. And all function invokation
 *        (cnrtInvokeFunction) will be performed on this channel too.
 *        Attention: The above policy only take effect when model parallelism
 *        is 1.
 *        This function is base on CPU thread context. So it's action scope
 *        is within current CPU thread. This function should be called after
 *        cnrtSetCurrentDevice;
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param cnrtChannelType_t[in] channel.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentChannel(cnrtChannelType_t channel);

/**
 * @brief Get current channel of current CPU thread.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pChannel[out] Pointer to channel.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetCurrentChannel(cnrtChannelType_t *pChannel);

/*********************************************************************************
 * Parameter descriptor related API
 *********************************************************************************/

/**
 * @brief Create parameter descriptor.
 *
 * @param param_desc[in] pointer to paramter descriptor.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateParamDesc(cnrtParamDesc_t *param_desc);

/**
 * @brief Destroy parameter descriptor.
 *
 * @param param_desc[in] paramter descriptor.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParamDesc(cnrtParamDesc_t param_desc);

/**
 * @brief Destroies cnrt param descriptor array.
 * @param param_descs[in] pointer of parameters.
 * @param param_num[in] length of parameters.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParamDescArray(cnrtParamDescArray_t param_descs,
                                                        int param_num);

/**
 * @brief Sets shape to cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dims[in] pointer of dim values.
 * @param dim_num[in] length of dims.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetShapeToParamDesc(cnrtParamDesc_t param_desc,
                                                      int *dims,
                                                      int dim_num);

/**
 * @brief Sets name to cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param name[in] pointer of name.
 * @param name_size[in] length of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetNameToParamDesc(cnrtParamDesc_t param_desc,
                                                     char *name,
                                                     int name_size);
/**
 * @brief Sets data type to cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dtype[in] data type of param.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetDataTypeToParamDesc(cnrtParamDesc_t param_desc,
                                                         cnrtDataType_t dtype);

/**
 * @brief Gets all dim product from cnrt param descriptor, can't contain dim less than 1.
 * @param param_desc[in] pointer of a parameter.
 * @param num[out] pointer of a num.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamElementNum(cnrtParamDesc_t param_desc, size_t *num);

/**
 * @brief Gets total size from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param size[out] pointer of size, is all shape multy datatype size,
 * shape should setted posstive integar.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamDescSize(cnrtParamDesc_t param_desc, int64_t *size);

/**
 * @brief Gets shape from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dims[out] pointer of dim values.
 * @param dim_num[out] length of dims.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetShapeFromParamDesc(cnrtParamDesc_t param_desc,
                                                        int **dims,
                                                        int *dim_num);

/**
 * @brief Gets name from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param name[out] pointer of name.
 * @param name_size[out] length of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetNameFromParamDesc(cnrtParamDesc_t param_desc,
                                                       char **name,
                                                       int *name_size);

/**
 * @brief Gets id from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param id[out] pointer of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetIdFromParamDesc(cnrtParamDesc_t param_desc, int *id);

/**
 * @brief Gets data type From cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dtype[out] data type of param.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDataTypeFromParamDesc(cnrtParamDesc_t param_desc,
                                                           cnrtDataType_t *dtype);

/**
 * @brief Creates cnrt param descriptor array.
 * @param param_descs[out] pointer of parameters.
 * @param param_num[in] length of parameters.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateParamDescArray(cnrtParamDescArray_t *param_descs,
                                                       int param_num);

/**
 * @brief Destroies cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParamDesc(cnrtParamDesc_t param_desc);

/**
 * @brief Destroies cnrt param descriptor array.
 * @param param_descs[in] pointer of parameters.
 * @param param_num[in] length of parameters.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyParamDescArray(cnrtParamDescArray_t param_descs,
                                                        int param_num);

/**
 * @brief Sets shape to cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dims[in] pointer of dim values.
 * @param dim_num[in] length of dims.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetShapeToParamDesc(cnrtParamDesc_t param_desc,
                                                      int *dims,
                                                      int dim_num);

/**
 * @brief Sets name to cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param name[in] pointer of name.
 * @param name_size[in] length of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetNameToParamDesc(cnrtParamDesc_t param_desc,
                                                     char *name,
                                                     int name_size);
/**
 * @brief Sets data type to cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dtype[in] data type of param.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetDataTypeToParamDesc(cnrtParamDesc_t param_desc,
                                                         cnrtDataType_t dtype);

/**
 * @brief Gets all dim product from cnrt param descriptor, can't contain dim less than 1.
 * @param param_desc[in] pointer of a parameter.
 * @param num[out] pointer of a num.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamElementNum(cnrtParamDesc_t param_desc, size_t *num);

/**
 * @brief Gets total size from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param size[out] pointer of size, is all shape multy datatype size,
 * shape should setted posstive integar.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParamDescSize(cnrtParamDesc_t param_desc, int64_t *size);

/**
 * @brief Gets shape from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dims[out] pointer of dim values.
 * @param dim_num[out] length of dims.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetShapeFromParamDesc(cnrtParamDesc_t param_desc,
                                                        int **dims,
                                                        int *dim_num);

/**
 * @brief Gets name from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param name[out] pointer of name.
 * @param name_size[out] length of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetNameFromParamDesc(cnrtParamDesc_t param_desc,
                                                       char **name,
                                                       int *name_size);

/**
 * @brief Gets id from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param id[out] pointer of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetIdFromParamDesc(cnrtParamDesc_t param_desc, int *id);

/**
 * @brief Gets data type From cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dtype[out] data type of param.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDataTypeFromParamDesc(cnrtParamDesc_t param_desc,
                                                           cnrtDataType_t *dtype);

/**
 * @brief Gets dim order from cnrt param descriptor.
 * @param param_desc[in] pointer of a parameter.
 * @param dim_order[out] data type of param.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDimOrderFromParamDesc(cnrtParamDesc_t param_desc,
                                                           cnrtDimOrder_t *dim_order);

/**
 * @brief Gets cnrt param descriptor from paramdesc array via param name.
 * @param param_desc[out] pointer of a parameter.
 * @param param_descs[in] pointer of parameter desc array.
 * @param param_num[in] number of parameter desc array.
 * @param name[in] pointer of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtGetParamDescFromParamDescArrayByName(cnrtParamDesc_t *param_desc,
                                         cnrtParamDescArray_t param_descs,
                                         int param_num,
                                         const char *name);

/**
 * @brief Gets cnrt param index from paramdesc array via param name.
 * @param param_desc[out] pointer of a parameter.
 * @param param_descs[in] pointer of parameter desc array.
 * @param param_num[in] number of parameter desc array.
 * @param name[in] pointer of name.
 * @return CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetIndexFromParamDescArrayByName(int *index,
                                                                   cnrtParamDescArray_t param_descs,
                                                                   int param_num,
                                                                   const char *name);

/**
 * @brief Reshape filter data from src address to dst address.
 *        The origin src data layout is src[N][H][W][C]
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dst[out] destination address.
 * @param src[in] source address.
 * @param n/h/w/c[in] the origin data layout.
 * @param type[in] the data type of dst[out] and src[in].
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtFilterReshape(void *dst, void *src, int n, int h, int w, int c, cnrtDataType_t type);

/**
 * @brief Reshape data from src address to dst address.
 *        only between NHWC and NCHW
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dst[out] destination address.
 * @param src[in] source address.
 * @param n/h/w/c[in] the origin data layout.
 * @param type[in] the data type of dst[out] and src[in].
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtReshapeNCHWToNHWC(void *dst, void *src, int n, int h, int w, int c, cnrtDataType_t type);

extern CNRT_DLL_API cnrtRet_t
cnrtReshapeNHWCToNCHW(void *dst, void *src, int n, int h, int w, int c, cnrtDataType_t type);

/**
 * @brief get model level from offline file.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param fname[in] offline file name.
 * @param model_level[out] model level.
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetModelLevelFromFile(const char *fname, int *model_level);

/****************************************************************************
 * Generic parameters handling
 ***************************************************************************/

/**
 * @brief Allocate a CNRT parameter context buffer
 *
 * @param pParam[out] pointer to the parameter context buffer pointer
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtAllocParam(void **pParam);

/**
 * @brief Destroy a CNRT parameter context buffer
 *
 * @param param[in] the parameter context buffer pointer
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestoryParam(void *param);

/**
 * @brief Add one parameter to parameter context buffer
 *
 * @param param[in] the parameter context buffer pointer
 * @param name[in] name of the parameter
 * @param len[in] length of the parameter
 * @param data[in] pointer to the parameter
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtAddParam(void *param, char *name, int len, void *data);

/**
 * @brief Get one parameter from parameter context buffer
 *
 * @param param[in] the parameter context buffer pointer
 * @param name[in] name of the parameter
 * @param out[out] result buffer
 * @param outlen[in] result buffer length
 *
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         CNRT_RET_ERR_MEMCPY if parameter actual length is larger than result buffer length
 *         CNRT_RET_ERR_NO_EXIST if "name" is not found in param context
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetParam(void *param, char *name, void *out, int outlen);

/**
 * @brief Convert a float/double to float16, store it at specific position (*f16 = (f16)d)
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param d[in] number to convert
 * @param f16[out] place to store
 * @retval error code of the last call of runtime functions
 */
extern CNRT_DLL_API cnrtRet_t cnrtConvertDoubleToHalf(uint16_t *f16, double x);

extern CNRT_DLL_API cnrtRet_t cnrtConvertFloatToHalf(uint16_t *f16, float d);

/**
 * @brief Convert a float16 to float/double, store it at specific position (*d =
 * (float/double)(f16))
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param f16[in] number to convert
 * @param d[out] place to store
 * @retval error code of the last call of runtime functions
 */

extern CNRT_DLL_API cnrtRet_t cnrtConvertHalfToDouble(double *d, uint16_t f16);

extern CNRT_DLL_API cnrtRet_t cnrtConvertHalfToFloat(float *d, uint16_t f16);

/**
 * @brief Get datatype's size.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dt[in] enum cnrtDataType variable.
 * @retval size of DataType,
 */
extern CNRT_DLL_API int cnrtDataTypeSize(cnrtDataType_t dt);

/**
 * @brief Create and deploy a runtime context on specified MLU device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[out] receiver pointer of runtime context
 * @param function[in] point to the MLU function. Function must be initialized from a compiled OP or
 *        from an offline model(cnrtExtractFunction)
 * @param extra[in]  Reserved for future use.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateRuntimeContext(cnrtRuntimeContext_t *pctx,
                                                       cnrtFunction_t function,
                                                       void *extra);

/**
 * This API is not recommended to use and will be deprecated in a next release.
 *
 * @brief Set channel on the specified MLU device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] receiver pointer of runtime context
 * @param channel[in] Assign the DDR channel of the runtime context.
 *        CNRT_CHANNEL_TYPE_NONE: Let CNRT decide channel. It is recommended for most users.
 *        CNRT_CHANNEL_TYPE_DUPLICATE: Const memory will be duplicated on DDR channels.
 *        It could improve concurrency performance when you have multiple threads or
 *        streams associating with this runtime context with the cost of memory consumption.
 *        For advanced users, you could assign channel manually.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetRuntimeContextChannel(cnrtRuntimeContext_t pctx,
                                                           cnrtChannelType_t channel);

/**
 * @brief Set device id on the specified MLU device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] receiver pointer of runtime context
 * @param dev_ordinal[in] The device ordinal of which the runtime context is deployed.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetRuntimeContextDeviceId(cnrtRuntimeContext_t pctx,
                                                            int dev_ordinal);

/**
 * @brief Initialize runtime context on the specified MLU device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer of runtime context.
 * @param extra[in] for expand.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInitRuntimeContext(cnrtRuntimeContext_t pctx, void *extra);

/**
 * @brief Create a runtime context queue.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer of runtime context.
 * @param queue[out] get a queue.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextCreateQueue(cnrtRuntimeContext_t pctx,
                                                            cnrtQueue_t *queue);

/**
 * @brief Create an event corresponding to a specified runtime context.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer of runtime context.
 * @param pnotifier[out] point to a notifier handle to retrieve newly created notifier.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextCreateNotifier(cnrtRuntimeContext_t pctx,
                                                               cnrtNotifier_t *pnotifier);

/**
 * @brief Allocate device memory by bytes array.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param ptr[out] point to the allocate memory array.
 * @param bytesArray[in] allocate memory size array.
 * @param num[in] allocate memory array length;
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextMallocBySizeArray(cnrtRuntimeContext_t pctx,
                                                                  void ***ptr,
                                                                  size_t *bytesArray,
                                                                  int length);

/**
 * @brief Free the memory space pointed by ptr, which must
 *        be returned by a previous call to cnrtRuntimeContextMemcpyByDesc.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer to runtime context
 * @param ptr[in] point to the memory to be free.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextFree(cnrtRuntimeContext_t pctx, void *ptr);

/**
 * @brief Free the memory space array pointed by ptr, which must
 *        be returned by a previous call to cnrtRuntimeContextMallocByDescArray.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer to runtime context
 * @param ptr[in] a pointer array.
 * @param length[in] array lenght.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtRuntimeContextFreeArray(cnrtRuntimeContext_t pctx,
                                                          void **ptr,
                                                          int length);

/**
 * @brief Destroy a runtime context.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer to runtime context
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyRuntimeContext(cnrtRuntimeContext_t pctx);

/**
 * We are going to support dynamic shape in the next release version(V4.1.0).
 * In order to avoid changing API, we expose dynamic shape API(
 * cnrtInvokeRuntimeContext_V2) in advance.
 * We strongly recommend you to use cnrtInvokeRuntimeContext_V2 rather than
 * cnrtInvokeRuntimeContext. @see cnrtInvokeRuntimeContext_V2.
 *
 * @brief Invoke a runtime context on MLU.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer to runtime context
 * @param params[in]  point to arguments.
 * @param queue[in] queue associated to the function call.
 * @param extra[in]  Reserved for future use.
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeRuntimeContext(cnrtRuntimeContext_t pctx,
                                                       void **params,
                                                       cnrtQueue_t queue,
                                                       void *extra);

/**
 * @brief Invoke a runtime context on MLU.
 *
 * We are going to support dynamic shape in the next release version(V4.1.0).
 * In order to avoid changing API, we expose dynamic shape API in advance.
 * In current release version(V4.0.0), you can pass NULL pointer to param_descs.
 * The behavior of cnrtInvokeRuntimeContext_V2 is the same as cnrtInvokeRuntimeContext.
 * We recommend you to use cnrtInvokeRuntimeContext_V2 rather than
 * cnrtInvokeRuntimeContext.
 *
 * @param pctx[in] pointer to runtime context
 * @param param_descs[in]  parameter descriptor array.
 * @param param_buffers[in] parameter buffer array.
 * @param queue[in] queue associated to the function call.
 * @param extra[in] reserved for future use.
 *
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInvokeRuntimeContext_V2(cnrtRuntimeContext_t pctx,
                                                          cnrtParamDesc_t *param_descs,
                                                          void **param_buffers,
                                                          cnrtQueue_t queue,
                                                          void *extra);

/**
 * @brief Get the runtime context info on the specified MLU device.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer of runtime context.
 * @param key[in] the key of the runtime context.
 * @param out[out] the value of the key.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetRuntimeContextInfo(cnrtRuntimeContext_t pctx,
                                                        cnrtRuntimeContextInfo_t key,
                                                        void **out);

/**
 * @brief Set current device to runtime context binded device
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param pctx[in] pointer of runtime context.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentContextDevice(cnrtRuntimeContext_t pctx);

/**
 * @brief Get the specific CPU bitmap according to the device index write to
 *        the struct DeviceAffinity
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param affinity[out] array reference in which to return a bitmask of CPUS, 64
 *        CPUS per unsigned long on 32 bit
 * @param dev_ordinal[in] the device dev_ordinal
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetDeviceAffinity(cnrtDeviceAffinity_t *affinity,
                                                    int dev_ordinal);

/**
 * @brief Clear the current thread affinity binding
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dev_ordinal[in] the device ordinal
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtClearCurrentThreadAffinity(int dev_ordinal);

/**
 * @brief Set the Current thread to the specific cpu according to the device affinity
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dev_ordinal[in] the device ordinal
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtSetCurrentThreadAffinity(int dev_ordinal);

/**
 * @brief Get the ordinal1 topology relationship with the ordinal2
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param relationship[out] the relationship of two device'topology
 * @param dev_ordinal1[in] the first device ordinal
 * @param dev_ordinal2[in] the second device ordinal
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtTopologyGetRelationship(cnrtTopologyRelationshipEnum_t *relationship,
                            int dev_ordinal1,
                            int dev_ordinal2);

/**
 * @brief Retrieve the set of devices that nearest to a given device at a specific
 *        interconnectivity level for all products
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param relationship[in] specified relationship
 * @param count[out] ordinalArray' size
 * @param ordinalArray[out] get the related devices's id
 * @param dev_ordinal[in] the device ordinal
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t
cnrtTopologyGetNearestDevices(cnrtTopologyRelationshipEnum_t relationship,
                              uint64_t *count,
                              uint64_t *ordinal_array,
                              int dev_ordinal);

/**
 * @brieif Retrieve the set of devices that have a CPU affinity with the given CPU number
 *         for all products
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param cpuid[in] specified cpu id
 * @param count[out] ordinalArray's size
 * @param ordinalArray[out] get related devices's id
 * @retval CNRT_RET_SUCCESS if success, otherwise with the error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtTopologyGetCpuRelatedDevices(int cpuid,
                                                               uint64_t *count,
                                                               uint64_t *ordinal_array);

/**
 * @brief Queries if a device(Dev) is capable of directly accessing memories on another(PeerDev).
 * @param CanPeer[out] Value to be returned. CanPeer is 1 represents Dev is of capable of directly
 *        accessing memories on PeerDev and 0 otherwise.
 * @param Dev[in] Deivce that directly accessing memories on another(PeerDev).
 * @param PeerDev[in] Deivce on which memories to be directly accessed by Dev.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetPeerAccessibility(unsigned int *CanPeer, int Dev, int PeerDev);

/**
 * @brief Copy memories from one device to another. The two devices should be peerable.
 *        You should set current device to srcDevice by calling cnrtSetCurrentDevice()
 *        before using this interface.
 * @param dst[in] Destination device memory pointer.
 * @param dstDevice[in] Destination device.
 * @param src[in] Source device memory pointer.
 * @param srcDevice[in] Source device.
 * @param count[in] Size of memory to be copied in bytes.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern cnrtRet_t cnrtMemcpyPeer(void *dst, int dstDevice, void *src, int srcDevice, size_t count);

/*
 * @brief Create the quantized param for cast data type.
 * @param param[out] pointer to cnrtQuantizedParam_t.
 * @param pos[in] the quantized value of position.
 * @param scale[in] the quantized value of scale.
 * @param offset[in] the quantized value of offset.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQuantizedParam(cnrtQuantizedParam_t *param,
                                                       int pos,
                                                       float scale,
                                                       int offset);

/*
 * @brief Create the quantized param for cast data type.
 * @param param[out] pointer to cnrtQuantizedParam_t.
 * @param poses[in] the quantized values of position.
 * @param scales[in] the quantized values of scale.
 * @param offsets[in] the quantized values of offset.
 * @param dimNum[in] the length of dimValues.
 * @param dimValues[in] the dim values of data to quant.
 * @param channelDim[in] the dim of channel in dim values.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCreateQuantizedParamByChannel(cnrtQuantizedParam_t *param,
                                                                int *poses,
                                                                float *scales,
                                                                float *offsets,
                                                                int dimNum,
                                                                int *dimValues,
                                                                int channelDim);

/*
 * @brief Destroy the quantized param.
 * @param param[in] pointer to cnrtQuantizedParam_t.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtDestroyQuantizedParam(cnrtQuantizedParam_t param);

/*
 * @brief Cast the data type from src address to dst address depend on param.
 *        if the param is null, no need quantized, support the cast data type:
 *        float32->float16, float32->uint8, int64->float16, float16->float32, float16->uint8,
 *        uint8->float32, uint8->float16, float32->float32
 *        if the parm is not null, need quantized, support the case data type:
 *        float32->float16, float32->int16, float32->int8, float32->int32, int32->float32,
 *        float16->int16, int16->float32, int8->float32, float32->float32
 * @param src_addr[in] pointer to src address.
 * @param src_data_type[in] the type of src data.
 * @param dst_addr[out] pointer to dst address.
 * @param dst_data_type[in] the type of dst data.
 * @param data_num[in] the number of need cast data.
 * @param param[in] pointer to cnrtQuantizedParam_t.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtCastDataType(void *src_addr,
                                               cnrtDataType_t src_data_type,
                                               void *dst_addr,
                                               cnrtDataType_t dst_data_type,
                                               int data_num,
                                               cnrtQuantizedParam_t param);

/*
 * @brief Add data stride when dst shape greater than src shape.
 * @param src_addr[in] pointer to src address.
 * @param data_type[in] pointer to cnrtDataType_t.
 * @param dst_addr[out] pointer to dst address.
 * @param dimNum[in] the number of dim.
 * @param dimValues[in] the values of dim array.
 * @param dimStride[in] the values of stride array which the specified dim need add
 *        specified stride sizes datas.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern CNRT_DLL_API cnrtRet_t cnrtAddDataStride(void *src_addr,
                                                cnrtDataType_t data_type,
                                                void *dst_addr,
                                                int dimNum,
                                                int *dimValues,
                                                int *dimStride);

/*
 * @brief Transform the data order to the op need by transform the order of the dim.
 * @param src_addr[in] pointer to src address.
 * @param data_type[in] pinter to cnrtDataType_t.
 * @param dst_addr[out] pointer to dst address.
 * @param dimNum[in] the num of dim.
 * @param dimValues[in] the values of dim array.
 * @param dimOrder[in] the values of dim array which dim order you want to transform.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern cnrtRet_t cnrtTransDataOrder(void *src_addr,
                                    cnrtDataType_t data_type,
                                    void *dst_addr,
                                    int dimNum,
                                    int dimValues[],
                                    int dimOrder[]);

/*
 * @brief Transform the data order and cast the data type.
 * @param src_addr[in] pointer to src address.
 * @param src_type[in] pinter to cnrtDataType_t of src.
 * @param dst_addr[out] pointer to dst address.
 * @param dst_type[in] pinter to cnrtDataType_t of dst.
 * @param param[in] pointer to cnrtQuantizedParam_t.
 * @param dimNum[in] the num of dim.
 * @param dimValues[in] the values of dim array.
 * @param dimOrder[in] the values of dim array which dim order you want to transform.
 * @retval CNRT_RET_SUCCESS if success. otherwise with error code.
 */
extern cnrtRet_t cnrtTransOrderAndCast(void *src_addr,
                                       cnrtDataType_t src_type,
                                       void *dst_addr,
                                       cnrtDataType_t dst_type,
                                       cnrtQuantizedParam_t param,
                                       int dimNum,
                                       int dimValues[],
                                       int dimOrder[]);
/**
 * @brief Get a series of input data sizes from a given function.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param sizeArray[out] point to the data size array.
 * @param num[out] length of the datasize array.
 * @param function[in] MLU function pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataSize(int64_t **sizeArray,
                                                   int *num,
                                                   cnrtFunction_t function);

/**
 * @brief Get a series of output data sizes from a given function.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param sizeArray[out] point to the data size array.
 * @param num[out] length of the datasize array.
 * @param function[in] MLU function pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataSize(int64_t **sizeArray,
                                                    int *num,
                                                    cnrtFunction_t function);

/**
 * @brief Get a series of input data type from a given function.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dtype[out] point to the input data type array.
 * @param num[out] length of the input data array.
 * @param function[in] MLU function pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataType(cnrtDataType_t **dtype,
                                                   int *num,
                                                   cnrtFunction_t function);

/**
 * @brief Get a series of output data type from a given function.
 *
 *  **Supports both MLU100 and MLU270**
 *
 * @param dtype[out] point to the output data type array.
 * @param num[out] length of the output array.
 * @param function[in] MLU function pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataType(cnrtDataType_t **dtype,
                                                    int *num,
                                                    cnrtFunction_t function);

/**
 * @brief Get the index input data shape from a given function.
 *
 *  **Support MLU270**
 *
 * @param dimValues[out] point to the input data dimValues array.
 * @param dimNum[out] length of the input data dimValues.
 * @param index[in] the index of input data.
 * @param function[in] MLU function pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetInputDataShape(int **dimValues,
                                                    int *dimNum,
                                                    int index,
                                                    cnrtFunction_t function);

/**
 * @brief Get the index output data shape from a given function.
 *
 *  **Supports MLU270**
 *
 * @param dimValues[out] point to the output data dimValues array.
 * @param dimNum[out] length of the output dimValues.
 * @param index[in] the index of output data.
 * @param function[in] MLU function pointer.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtGetOutputDataShape(int **dimValues,
                                                     int *dimNum,
                                                     int index,
                                                     cnrtFunction_t function);
/**
 * @brief Inference output shape by input_param, func should be inited from model or fusion_op,
 * inputshape should set to input_params. This func will fill datatype and dim_order to params if
 * get right shape.
 *
 *  **Supports MLU270.**
 *
 * @param func[in] pointer of cnrt function.
 * @param input_num[in] num of input paramdescs.
 * @param input_params[in] pointer of input paramdescs.
 * @param output_num[in] num of output paramdescs.
 * @param output_params[in] pointer of output paramdescs.
 * @retval CNRT_RET_SUCCESS if success,
 *         otherwise the error code is returned.
 */
extern CNRT_DLL_API cnrtRet_t cnrtInferFunctionOutputShape(cnrtFunction_t func,
                                                           int input_num,
                                                           cnrtParamDescArray_t input_params,
                                                           int output_num,
                                                           cnrtParamDescArray_t output_params);

/*! A pointer which points to void */
typedef void *cnrtIpcMemHandle;

/*
 * @brief Acquire an interprocess memory handle for an existing device memory allocation.
 *        cnrtSetCurrentDevicie() should be called before using this interface.
 *
 * @param handle[out] The unique handle for device memory share.
 * @param devPtr[in] Base pointer to previously allocated device memory.
 * @retval CNRT_RET_SUCESS if success, otherwise with the error code.
 */
extern cnrtRet_t cnrtAcquireMemHandle(cnrtIpcMemHandle *handle, void *devPtr);

/*
 * @brief Map an interprocess memory handle exported from another process and returns the device
 *        memory pointer usable in the local process. cnrtSetCurrentDevicie() should be called
 * before using this interface.
 *
 * @param devPtr[out] Returned device memory pointer.
 * @param handle[in] The unique handle for device memory to map.
 * @param flag[in] Flag for this operation. 0 is Reserved.
 * @retval CNRT_RET_SUCESS if success, otherwise with the error code.
 */
extern cnrtRet_t cnrtMapMemHandle(void **devPtr, cnrtIpcMemHandle handle, int flag);

/*
 * @brief Unmap memory that mapped with cnrtMapMemHandle.
 *        cnrtSetCurrentDevicie() should be called before using this interface.
 *
 * @param devPtr[in] Device memory pointer.
 * @retval CNRT_RET_SUCESS if success, otherwise with the error code.
 */
extern cnrtRet_t cnrtUnMapMemHandle(void *ptr);

#if defined(__cplusplus)
}
#endif /*__cplusplus*/
#endif /*__CNRT_H*/
