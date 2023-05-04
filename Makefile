download_dataset:
	mkdir -p ./dataset
	curl 'https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/all-samples.zip' -o ./dataset/dataset.zip
	python3 unpackdataset.py

clean_dataset:
	rm -rf ./dataset

process_dataset:
	# following files can not be converted so we delete them from dataset
	rm dataset/all-samples/saxophone_Fs3_15_fortissimo_normal.mp3
	rm dataset/all-samples/viola_D6_05_piano_arco-normal.mp3
	python3 convertsample.py
	python3 resampling.py
	python3 mcff.py

reset_dataset: clean_dataset download_dataset process_dataset

split_chunks:
	rm -rf dataset/Split1
	rm -rf dataset/Split2
	rm -rf dataset/Split3
	python3 analyzedataset2.py
	python3 split_files.py 5 20 20