# Usage:
# bash scripts/extract_contours.sh

source env/bin/activate

echo "Erk random"
python -m src.clusterability.precompute erk-random
echo "Erk phrase"
python -m src.clusterability.precompute erk-phrase
echo "Boehme random"
python -m src.clusterability.precompute boehme-random
echo "Boehme phrase"
python -m src.clusterability.precompute boehme-phrase
echo "Creighton random"
python -m src.clusterability.precompute creighton-random
echo "Creighton phrase"
python -m src.clusterability.precompute creighton-phrase

# echo "Han random"
# python -m src.clusterability.precompute han-random
echo "Han phrase"
python -m src.clusterability.precompute han-phrase
# echo "Natmin random"
# python -m src.clusterability.precompute natmin-random
echo "Natmin phrase"
python -m src.clusterability.precompute natmin-phrase
# echo "Shanxi random"
# python -m src.clusterability.precompute shanxi-random
echo "Shanxi phrase"
python -m src.clusterability.precompute shanxi-phrase

# echo "Antiphons random"
# python -m src.clusterability.precompute liber-antiphons-random
echo "Antiphons phrase"
python -m src.clusterability.precompute liber-antiphons-phrase
# echo "Responsories random"
# python -m src.clusterability.precompute liber-responsories-random
echo "Responsories phrase"
python -m src.clusterability.precompute liber-responsories-phrase
# echo "Alleluias random"
# python -m src.clusterability.precompute liber-alleluias-random
echo "Alleluias phrase"
python -m src.clusterability.precompute liber-alleluias-phrase

echo "Cantus antiphons neumes"
python -m src.clusterability.precompute cantus-antiphon-neumes
echo "Cantus antiphons syllables"
python -m src.clusterability.precompute cantus-antiphon-syllables
echo "Cantus antiphons words"
# python -m src.clusterability.precompute cantus-antiphon-words
# echo "Cantus antiphons poisson-3"
# python -m src.clusterability.precompute cantus-antiphon-poisson-3
# echo "Cantus antiphons poisson-5"
# python -m src.clusterability.precompute cantus-antiphon-poisson-5
# echo "Cantus antiphons poisson-7"
# python -m src.clusterability.precompute cantus-antiphon-poisson-7

echo "Cantus responsories neumes"
python -m src.clusterability.precompute cantus-responsory-neumes
echo "Cantus responsories syllables"
python -m src.clusterability.precompute cantus-responsory-syllables
echo "Cantus responsories words"
python -m src.clusterability.precompute cantus-responsory-words
# echo "Cantus responsories poisson-3"
# python -m src.clusterability.precompute cantus-responsory-poisson-3
# echo "Cantus responsories poisson-5"
# python -m src.clusterability.precompute cantus-responsory-poisson-5
# echo "Cantus responsories poisson-7"
# python -m src.clusterability.precompute cantus-responsory-poisson-7