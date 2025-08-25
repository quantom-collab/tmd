CXX = clang++

# Basic compiler flags
CXXFLAGS = -O3 -fPIC -std=c++11

# NANGAPARBAT
NANGAPARBATINCS = $(shell NangaParbat-config --cppflags)
NANGAPARBATLIBS = $(shell NangaParbat-config --ldflags)

# APFEL++
APFELPPINCS = $(shell apfelxx-config --cppflags)
APFELPPLIBS = $(shell apfelxx-config --ldflags)

# LHAPDF
LHAPDFINCS = $(shell lhapdf-config --cppflags)
LHAPDFLIBS = $(shell lhapdf-config --ldflags)

# YAML
YAMLINCS = $(shell pkg-config --cflags yaml-cpp)
YAMLLIBS = $(shell pkg-config --libs   yaml-cpp)

# GSL
GSLINCS = -I/opt/homebrew/opt/gsl/include
GSLLIBS = -L/opt/homebrew/opt/gsl/lib -lgsl -lgslcblas

# Now set up the compiler and link flags and libs
CXXFLAGS += $(NANGAPARBATINCS) $(APFELPPINCS) $(LHAPDFINCS) $(YAMLINCS) $(GSLINCS)
LDFLAGS  += $(NANGAPARBATINCS) $(APFELPPINCS) $(LHAPDFINCS)
CLIBS += $(NANGAPARBATLIBS) $(APFELPPLIBS) $(LHAPDFLIBS) $(YAMLLIBS) $(GSLLIBS)

install : all
all : CollinsSoperKernel GridTMDbT

CollinsSoperKernel: CollinsSoperKernel.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(CLIBS)

GridTMDbT: GridTMDbT.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(CLIBS)

.SUFFIXES : .cc .o .f .c

.cc.o:
	$(CXX) $(CXXFLAGS) -c $<

.f.o:
	$(F77)  -c $<

clean:
	rm -rf *.lo *.o *.la CollinsSoperKernel GridTMDbT *~
