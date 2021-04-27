run: tag
	mkdir -p mud_figures/
	docker run --rm -ti -v $(shell pwd)/mud_figures:/work mudex

build: bin/Dockerfile
	docker build -t mudex -f bin/Dockerfile \
	  --build-arg USER_ID=$(shell id -u) \
	  --build-arg GROUP_ID=$(shell id -g) .

tag: build
	docker tag mudex mathematicalmichael/mudex:$(shell date +"%Y%m%d")
	docker tag mudex mathematicalmichael/mudex:latest

push: tag
	docker push mathematicalmichael/mudex:$(shell date +"%Y%m%d")
	docker push mathematicalmichael/mudex:latest

version:
	./bin/dmud.sh mud_examples --version

clean:
	rm -rf src/mud_examples/.ipynb_checkpoints
	rm -rf mud_figures/