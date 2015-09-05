#!/bin/bash

for FILE in `find . -name "*.eps"`; do
	epstopdf "${FILE}"
	pdfcrop "${FILE%.eps}.pdf" "${FILE%.eps}.pdf"
	rm "${FILE}"
done


