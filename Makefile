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
	KTRANSFORMERS_FORCE_BUILD=TRUE pip install -e . -v --no-build-isolation
	echo "Installation completed successfully"
clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf ktransformers/ktransformers_ext/build
	rm -rf ktransformers/ktransformers_ext/cuda/build
	rm -rf ktransformers/ktransformers_ext/cuda/dist
	rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info	
install_numa:
	USE_NUMA=1 make dev_install
install_no_numa:
	env -u USE_NUMA make dev_install
