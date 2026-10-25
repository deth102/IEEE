/* Instantiates the generated model constants and tables (single translation unit).
 * fd_model_data.h lives in firmware/models/<tag>/, which must be on the include path. */
#include "fd_core.h"
#include "fd_model_data.h"

#if defined(__GNUC__) && !defined(_WIN32)
/* default timer: platforms override with a strong definition (cycle counter) */
__attribute__((weak)) uint32_t fd_cycles(void) { return 0; }
#endif
