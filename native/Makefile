GCC_OPTS = -Wl,--add-stdcall-alias -O3

all : loess.dll

loess.dll : Loess.o
	gcc $(GCC_OPTS) -shared -o loess.dll Loess.o

Loess.o : 
	gcc -I"$(JAVA_HOME)\include" -I"$(JAVA_HOME)\include\win32" $(GCC_OPTS) -c Loess.c -o Loess.o

clean :
	del Loess.o loess.dll