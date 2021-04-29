help:
	docker run --rm -i mudex mud_examples --version
	docker run --rm -i mudex mud_examples --help

test:
	docker run --rm -i mudex

all:
	@echo "Running all examples with default entrypoint"
	./bin/dmud.sh

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

clean:
	rm -rf src/mud_examples/.ipynb_checkpoints
	rm -rf mud_figures/
