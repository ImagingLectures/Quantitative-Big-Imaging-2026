#!/bin/bash

if command -v pixi &> /dev/null && pixi run which "jupyter-book" &> /dev/null; then
    pixi run jupyter-book build Lectures/Lecture-09 --builder pdflatex
else
    jupyter-book build Lectures/Lecture-09 --builder pdflatex
fi

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=docs/QBI-Lecture09-DynamicExperiments.pdf Lectures/Lecture-09/_build/latex/QBI-Lecture09-DynamicExperiments.pdf
