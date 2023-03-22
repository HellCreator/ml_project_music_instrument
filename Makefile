download_dataset:
	mkdir -p ./dataset
	curl 'https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/all-samples.zip' -o ./dataset/dataset.zip
	python3 unpackdataset.py

clean_dataset:
	rm -rf ./dataset
