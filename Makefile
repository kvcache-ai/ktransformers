flake_find:
	cd ktransformers && flake8 | grep -Eo '[A-Z][0-9]{3}' | sort | uniq| paste -sd ',' - 
format:
	@cd ktransformers && black .
	@black setup.py
dev_install:
# clear build dirs
	rm -rf build
	rm -rf *.egg-info
	rm -rf ktransformers/ktransformers_ext/build
	rm -rf ktransformers/ktransformers_ext/cuda/build
	rm -rf ktransformers/ktransformers_ext/cuda/dist
	rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info

# install ktransformers
	echo "Installing python dependencies from requirements.txt"
	pip install -r requirements-local_chat.txt

	echo "Installing ktransformers"
	KTRANSFORMERS_FORCE_BUILD=TRUE pip install -e . --no-build-isolation
	echo "Installation completed successfully"