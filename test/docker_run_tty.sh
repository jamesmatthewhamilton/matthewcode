#!/bin/sh
# Mock of a wrapper script around `docker run -it ...`. Fails exactly the way
# docker does when no terminal is attached, without needing docker installed.
if [ ! -t 0 ] || [ ! -t 1 ]; then
    echo "the input device is not a TTY" >&2
    exit 1
fi
echo "hello from container"
