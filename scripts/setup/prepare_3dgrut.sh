#!/usr/bin/env bash
cd ThirdParty
if [ ! -d "3dgrut" ]; then
    git clone https://github.com/r4hul77/3dgrut.git
fi
cd 3dgrut
git checkout main
git pull
cd ..
cd ..
