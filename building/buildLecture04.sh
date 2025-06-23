#!/bin/bash

if command -v pixi &> /dev/null && pixi run which "jupyter-book" &> /dev/null; then
    pixi run jupyter-book build Lectures/Lecture-04 --builder pdflatex
else
    jupyter-book build Lectures/Lecture-04 --builder pdflatex
fi

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=docs/QBI-Lecture04-BasicSegmentation.pdf Lectures/Lecture-04/_build/latex/QBI-Lecture04-BasicSegmentation.pdf
