default_target: all
.PHONY : default_target debug

CC = g++
STD = 11 -g
PROG = main
RELOBJS = ./relobj/main.o
DEBOBJS = ./debobj/main.o 

all: $(RELOBJS)
	$(CC) -std=c++$(STD) $(RELOBJS) -o $(PROG)

debug: $(DEBOBJS)
	$(CC) -std=c++$(STD) $(DEBOBJS) -o $(PROG)

./relobj/main.o: main.cpp simple_matrix.h NN.h
	$(CC) -c -O2 -std=c++$(STD) main.cpp -o ./relobj/main.o  

./debobj/main.o: main.cpp simple_matrix.h NN.h
	$(CC) -c -O0 -D DEBUG -std=c++$(STD) main.cpp -o ./debobj/main.o


clean:
	rm -r ./relobj ./debobj $(PROG)

$(shell   mkdir -p relobj)
$(shell   mkdir -p debobj)
