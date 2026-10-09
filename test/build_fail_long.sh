#!/bin/sh
# Mock of a long build that always fails. The real cause is one line buried in
# the middle of tens of thousands of lines of output; the tail is only a
# cascade of follow-on failures, exactly like a big C++ build.
i=1
while [ "$i" -le 30000 ]; do
    echo "[ $i/60000 ] Building CXX object src/module_$i.cpp.o"
    i=$((i + 1))
done
# On stdout, as ninja/cmake report captured compiler output, so it is truly
# buried in both piped and tty modes.
echo "Error: The solution is simple!"
while [ "$i" -le 60000 ]; do
    echo "[ $i/60000 ] Building CXX object src/module_$i.cpp.o"
    i=$((i + 1))
done
j=1
while [ "$j" -le 200 ]; do
    echo "ld: undefined reference to symbol_$j" >&2
    j=$((j + 1))
done
exit 1
