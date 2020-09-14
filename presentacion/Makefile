all: diapositivas.pdf

diapositivas.pdf: diapositivas.tex
	xelatex --shell-escape $^

clean:
	rm -f diapositivas.aux diapositivas.log diapositivas.out diapositivas.nav diapositivas.vrb diapositivas.snm

open:
	evince diapositivas.pdf &
