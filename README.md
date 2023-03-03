Shapes of music
===============

This repository contains data and code for the paper 'Shapes of music: Contour typology revisited', which is currently being prepared for publication.

---


<img src="figures/synthetic-contours/synthetic-contours.jpg?raw=true" width="800" 
    title="Three approaches to mode classification in plainchant compared">


**Abstract.** How can one best describe the shapes of melodic phrases in musics from across the globe? Previous studies have often relied on typologies using a discrete set of contour types. We question their adequacy, as we find no evidence that phrase contours cluster into discrete types, in neither German and Chinese folksongs, nor in Gregorian chant. The test for clustering we propose applies the dist-dip test of multimodality after a UMAP dimensionality reduction. The test correctly identifies clustering in a synthetic dataset of contours, but not in actual phrase contours. These results argue against the use of discrete typologies. Additionally, we identify a hidden parameter in two discrete typologies that can strongly skew the type distributions. Taken together, our findings suggest that melodic contour is best seen as a continuous phenomenon. We end by revisiting the melodic arch hypothesis using a continuous approach to contour.

---

Repository structure
-------------------

To do 


Python setup
------------

You can find the Python version used in .python-version and all dependencies are listed in `requirements.txt`. If you use pyenv and venv to manage python versions and virtual environments, do the following:

```bash
# Install the right python version
pyenv install | cat .python-version

# Create a virtual environment
python -m venv env

# Activate the environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

