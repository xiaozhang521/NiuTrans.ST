#ifndef __UTILS__
#define __UTILS__

#include <stdint.h>
#include <iostream>
namespace s2t {

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244008443621048490
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifndef COMPLEXFFT_BLOCKSIZE
#define COMPLEXFFT_BLOCKSIZE 8192
#endif // !COMPLEXFFT_BLOCKSIZE


#define SWAP8(a) do { \
      int t = (reinterpret_cast<char*>(&a))[0];\
              (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[7];\
              (reinterpret_cast<char*>(&a))[7]=t;\
          t = (reinterpret_cast<char*>(&a))[1];\
              (reinterpret_cast<char*>(&a))[1]=(reinterpret_cast<char*>(&a))[6];\
              (reinterpret_cast<char*>(&a))[6]=t;\
          t = (reinterpret_cast<char*>(&a))[2];\
              (reinterpret_cast<char*>(&a))[2]=(reinterpret_cast<char*>(&a))[5];\
              (reinterpret_cast<char*>(&a))[5]=t;\
          t = (reinterpret_cast<char*>(&a))[3];\
              (reinterpret_cast<char*>(&a))[3]=(reinterpret_cast<char*>(&a))[4];\
              (reinterpret_cast<char*>(&a))[4]=t;} while (0)
#define SWAP4(a) do { \
      int t = (reinterpret_cast<char*>(&a))[0];\
              (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[3];\
              (reinterpret_cast<char*>(&a))[3]=t;\
          t = (reinterpret_cast<char*>(&a))[1];\
              (reinterpret_cast<char*>(&a))[1]=(reinterpret_cast<char*>(&a))[2];\
              (reinterpret_cast<char*>(&a))[2]=t;} while (0)
#define SWAP2(a) do { \
      int t = (reinterpret_cast<char*>(&a))[0];\
              (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[1];\
              (reinterpret_cast<char*>(&a))[1]=t;} while (0)
    /*
#define ASSERT(cond) do {                                                                         \
        if (cond)                                                                  \
          (void)0;                                                                 \
        else                                                                       \
          std::cout << "Assert ERROR";       \
      } while (0)
      */
#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif
#ifndef __FILENAME
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )
#endif
#ifndef ASSERT
#define ASSERT(cond) do { \
        if (cond) \
            (void)0; \
        else \
        { \
            fprintf(stderr, "[ERROR] calling '%s' in %s\n", #cond, __FILENAME__ ); \
            throw; \
        } \
    }while(0)
#endif // !ASSERT


}

#endif // !__UTILS__


