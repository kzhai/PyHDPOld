PyHDP
=====

Generate data

	python util/generate_data.py --number_of_topics=5 --number_of_vocabularies=20 --number_of_documents=200 --output_directory=../input/

Profile sampler

	python -m cProfile hdp/launch_profiler.py
