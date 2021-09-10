.PHONY: all

all: linux_requirements python_requirements MFA_thirdparty MFA_pretrained

linux_requirements:
	sudo apt-get update && sudo apt-get install -y libsndfile1 libopenblas-dev

python_requirements:
	conda install -c conda-forge -y --file conda_requirements.txt
	pip install pip setuptools --upgrade
	pip install -e ../.[pytorch] \
		--find-links https://download.pytorch.org/whl/torch_stable.html

MFA_thirdparty:
	mfa thirdparty download
	mfa thirdparty validate

MFA_pretrained:
	mfa download acoustic english
	mfa download g2p english_g2p
	mfa download dictionary english
