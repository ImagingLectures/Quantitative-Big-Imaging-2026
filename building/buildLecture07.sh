#!/bin/bash

unzip Lectures/Lecture-07/data/grains.npy.zip
unzip Lectures/Lecture-07/data/ws_grains.npy.zip
unzip Lectures/Lecture-07/data/Cropped_prediction_8bit.npy.zip

if command -v pixi &> /dev/null && pixi run which "jupyter-book" &> /dev/null; then
    pixi run jupyter-book build Lectures/Lecture-07 --builder pdflatex
else
    jupyter-book build Lectures/Lecture-07 --builder pdflatex
fi

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=docs/QBI-Lecture07-ComplexShape.pdf Lectures/Lecture-07/_build/latex/QBI-Lecture07-ComplexShape.pdf