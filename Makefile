build:
	mkdir ./dist
	cp ./text_reuse_pipline/main.py ./dist
	zip -x main.py -r ./dist/text_reuse_pipeline.zip text_reuse_pipline/