# CFLAGS += $(shell pkg-config --cflags hwloc)
# CFLAGS += -march=armv8.2-a+fp16+dotprod+sve+bf16 -I/home/test/kt-code/HPCKit_25.0.0_Linux-aarch64/package/KunpengHPCKit-kml.25.0.0/include
# CFLAGS += -march=armv8.2-a+fp16+dotprod+sve+bf16 -I/home/test/kt-code/HPCKit_25.0.0_Linux-aarch64/package/KunpengHPCKit-kml.25.0.0/include
CFLAGS += -O3
CFLAGS += -I/usr/local/include/blis/
LDLIBS += -L/usr/local/lib -lblis
# LDLIBS += $(shell pkg-config --libs hwloc) -lkml_rt

CXX = g++

# i8_cal: i8_cal.cpp
# $(CXX) i8_cal.cpp $(CFLAGS) -o i8_cal $(LDLIBS)
# run: i8_cal
# ./i8_cal

simple_test_build: simple_test.cpp
	rm -f simple_test
	BLAS_NUM_THREADS=1 $(CXX) simple_test.cpp $(CFLAGS) -o simple_test $(LDLIBS)

simple_aocl_build: simple_test_aocl.cpp
	rm -f simple_test_aocl
	BLAS_NUM_THREADS=1 $(CXX) simple_test_aocl.cpp $(CFLAGS) -o simple_test_aocl $(LDLIBS)

fp16_test_build: fp16-test.cpp
	rm -f fp16-test
	$(CXX) fp16-test.cpp $(CFLAGS) -o fp16-test $(LDLIBS)
bf16_test_build: bf16-test.cpp
	rm -f bf16-test
	$(CXX) bf16-test.cpp $(CFLAGS) -o bf16-test $(LDLIBS)

run: simple_aocl_build
	LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH  ./simple_test_aocl