run:
	mkdir -p mud_figures/ && \
	docker run --rm -ti -v $(shell pwd)/mud_figures:/work mud

build: bin/Dockerfile
	docker build -t mud -f bin/Dockerfile \
	  --build-arg USER_ID=$(shell id -u) \
	  --build-arg GROUP_ID=$(shell id -g) .

tag: build
	docker tag mud mathematicalmichael/mud:$(shell date +"%Y%m%d")
	docker tag mud mathematicalmichael/mud:latest

push: tag
	docker push mathematicalmichael/mud:$(shell date +"%Y%m%d")
	docker push mathematicalmichael/mud:latest

clean:
	rm -rf src/mud_examples/.ipynb_checkpoints
	rm -rf mud_figures/
