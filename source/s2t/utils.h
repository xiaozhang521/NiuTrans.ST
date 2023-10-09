#ifndef __UTILS__
#define __UTILS__

#include <stdint.h>
#include <iostream>
namespace s2t {

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

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
#define ASSERT(cond) do {                                                                         \
        if (cond)                                                                  \
          (void)0;                                                                 \
        else                                                                       \
          std::cout << "Assert ERROR";       \
      } while (0)


}

#endif // !__UTILS__


