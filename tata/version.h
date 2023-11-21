#pragma once
#include "svn_version.h"

/////////////////////////////// 추가
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)

#define VERSION_MAJOR               1
#define VERSION_MINOR               0
#define VERSION_BUG_FIX             0
#define VERSION_BUILD				SVN_REVISION


#if SVN_LOCAL_MODIFICATIONS
#define VERSION_MODIFIER            "M"
#else
#define VERSION_MODIFIER
#endif

#define VER_FILE_VERSION            VERSION_MAJOR, VERSION_MINOR, VERSION_BUG_FIX, VERSION_BUILD

#define VER_FILE_VERSION_STR        STRINGIZE(VERSION_MAJOR)        \
                                    "." STRINGIZE(VERSION_MINOR)   \
									"." STRINGIZE(VERSION_BUG_FIX)    \
									"." STRINGIZE(VERSION_BUILD) \
/*
#define VER_FILE_VERSION_STR_NO_BUILD        STRINGIZE(VERSION_MAJOR)        \
                                    "." STRINGIZE(VERSION_MINOR)     \                                   
                                    "." STRINGIZE(VERSION_BUG_FIX)     \
									//"." STRINGIZE(VERSION_BUILD) \
*/
#define VER_COMPANYNAME_STR         "Medical IP"
#define VER_FILEDESCRIPTION_STR     "TiSepX tata module dll"
#define VER_INTERNALNAME_STR        "TiSepXTataAPI"
#define VER_LEGALCOPYRIGHT_STR      "Copyright (c) Medical IP Corp."
#define VER_LEGALTRADEMARKS1_STR    "All Rights Reserved"
#define VER_LEGALTRADEMARKS2_STR    VER_LEGALTRADEMARKS1_STR

#define VER_ORIGINAL_FILENAME_STR	"tata.dll"										
#define VER_PRODUCT_NAME_STR		"TiSepX"
#define VER_PRODUCT_VERSION_STR		"1.0.4.0"

