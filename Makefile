build:
	docker build -t mud -f Dockerfile .

tag: build
	docker tag mud mathematicalmichael/mud:$(shell date +"%Y%m%d")
	docker tag mud mathematicalmichael/mud:latest

push: tag
	docker push mathematicalmichael/mud:$(shell date +"%Y%m%d")
	docker push mathematicalmichael/mud:latest
clean:
	rm -rf src/mud_examples/.ipynb_checkpoints
