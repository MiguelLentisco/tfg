all: compile

compile:
	latexmk -pdf libro.tex

open:
	evince libro.pdf &

clean:
	find . -name "*.aux" -type f -delete
	rm -f libro.toc libro.out libro.lot libro.log libro.lof libro.ind libro.ilg libro.fls libro.fdb_latexmk libro.brf libro.blg libro.bbl libro.idx
