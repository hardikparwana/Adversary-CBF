#!/bin/bash
cmake -DCMAKE_PREFIX_PATH=/home/hardik/softwares/libtorch/ ..
cmake --build . --config Release
