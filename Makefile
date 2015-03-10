CC = g++ 
CLCC = openclcc
CFLAGS = -Wall -fbounds-check
INCS = -I. -I${ATISTREAMSDKROOT}/include
LIBS = -lOpenCL
OBJS = main.o

all: cl_test 

cl_test: kernel $(OBJS)
	$(CC) $(CFLAGS) $(INCS) -o $@ $(OBJS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@

kernel: $(wildcard *.cl)
	$(CLCC) $<

clean:
	rm -f *.o *.x *.ll cl_test
