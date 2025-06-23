#!/bin/bash

if command -v pixi &> /dev/null && pixi run which "jupyter-book" &> /dev/null; then
    pixi run jupyter-book build Lectures/Lecture-03 --builder pdflatex
else
    jupyter-book build Lectures/Lecture-03 --builder pdflatex
fi

gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=docs/QBI-Lecture03-ImageEnhancement.pdf Lectures/Lecture-03/_build/latex/QBI-Lecture03-ImageEnhancement.pdf